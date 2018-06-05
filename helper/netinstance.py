import torch
import torch.optim as optim
from torch.autograd import Variable
from model.net import UnsupervisedHomographyNet
import time
from torch.utils.data import DataLoader
import os, cv2, sys, numpy, signal

def l2_corner_loss(output, target):
    l2 = output-target
    l2 = l2.pow(2)
    l2 = l2.sum()
    return l2 / 2

class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self,signum, frame):
    self.kill_now = True

class NetInstance:
    def __init__(self, learning_rate=0.0001, input_file=None, net_name=None, map_name="homographymap"):

        self.killer = GracefulKiller()

        self.learning_rate=learning_rate
        self.input_file = input_file
        self.patch_size = (128, 128)
        self.patch_location = (128, 128)

        self.start_epoch=0

        self.running_loss = []
        self.ground_truth_loss = []
        self.validation_loss = []
        self.ground_truth_validation_loss = []
        self.extra_validation_loss = []
        self.state_dict = None
        self.optimizer_state_dict = None

        self.net_name=net_name
        self.map_name=map_name

        self.last_evaluation_loss = None

        self.read_checkpoint()

        if self.net_name is None:
            raise Exception('No net name in arguments or checkpoint - exit')
        if self.map_name is None:
            raise Exception('No map name in arguments or checkpoint - exit')
        self.net = UnsupervisedHomographyNet(self.patch_size, self.patch_location, self.net_name, self.map_name)

        self.net.cuda()
        if self.state_dict is not None:
            self.net.load_state_dict(self.state_dict)
        self.optimizer = optim.Adam(self.net.parameters(), self.learning_rate)
        if self.optimizer_state_dict is not None:
            self.optimizer.load_state_dict(self.optimizer_state_dict)

        if len(self.extra_validation_loss) < self.start_epoch:
            self.extra_validation_loss.extend([0 for i in range(len(self.extra_validation_loss), self.start_epoch)])

    def read_checkpoint(self, net=None):

        if self.input_file is not None and os.path.isfile(self.input_file):
            print("=> loading checkpoint '{}'".format(self.input_file))
            checkpoint = torch.load(self.input_file)
            self.start_epoch = checkpoint['epoch']
            self.running_loss = checkpoint['loss']
            self.validation_loss = checkpoint['val']
            self.state_dict = checkpoint['state_dict']
            self.optimizer_state_dict = checkpoint['optimizer']

            if "ground_truth_loss" in checkpoint and "ground_truth_validation_loss" in checkpoint:
                self.ground_truth_loss = checkpoint['ground_truth_loss']
                self.ground_truth_validation_loss = checkpoint['ground_truth_validation_loss']
                self.has_ground_truth = True
            else:
                self.has_ground_truth = False

            if "netargs" in checkpoint:
                print "Net arguments found..."

                raise Exception('Net arguments are not compatible anymore...')

            if "net_name" in checkpoint:
                if self.net_name != checkpoint["net_name"]:
                    print "Argument net name is overriden by checkpoint net name: is now "+checkpoint["net_name"]

                self.net_name=checkpoint["net_name"]

            if "map_name" in checkpoint:
                if self.map_name != checkpoint["map_name"]:
                    print "Argument map name is overriden by checkpoint net name: is now "+checkpoint["map_name"]

                self.map_name=checkpoint["map_name"]

            if "extra_val" in checkpoint:
                self.extra_validation_loss = checkpoint["extra_val"]

            #else:
                #self.netargs = None
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(self.input_file, checkpoint['epoch']))

            for i, loss in enumerate(self.running_loss):
                print("Epoch {}: loss = {}".format(i + 1, self.running_loss[i]))

        else:
            print 'no checkpoint found - begin from beginning'

    def save_checkpoint(self, epoch, output_file, has_ground_truth):
        if output_file is not None:

            if has_ground_truth:
                checkpoint = {
                    'epoch': epoch + 1,
                    'loss': self.running_loss,
                    'val': self.validation_loss,
                    'state_dict': self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    #'netargs': self.netargs,
                    'ground_truth_loss': self.ground_truth_loss,
                    'ground_truth_validation_loss': self.ground_truth_validation_loss,
                    'net_name':self.net_name,
                    'map_name':self.map_name
                }
            else:
                checkpoint = {
                    'epoch': epoch + 1,
                    'loss': self.running_loss,
                    'val': self.validation_loss,
                    'state_dict': self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    #'netargs': self.netargs,
                    'net_name': self.net_name,
                    'map_name': self.map_name
                }

            if len(self.extra_validation_loss) > 0:
                checkpoint["extra_val"] = self.extra_validation_loss

            torch.save(checkpoint, output_file)

            print('Saved')
        else:
            print "No output file, not saved!"

    def get_loss(self):

        return self.running_loss, self.validation_loss, self.ground_truth_loss, self.ground_truth_validation_loss, self.extra_validation_loss


    def train_and_validate(self, dataset, batch_size=40, validation_batch_size=20, validation_dataset=None, extra_validation_dataset=None, output_file=None, nb_epochs=None, show_first_image=False, supervised=False):

        if dataset.has_ground_truth():
            if supervised:
                print "Supervised training - using ground truth data"
            else:
                print "Unsupervised training without ground truth data"
        else:
            print "Dataset without ground truth data, no ground truth training or validation"
            supervised = False

        if nb_epochs is None:
            nb_epochs = 1000000
        else:
            nb_epochs = nb_epochs

        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=True, num_workers=4)



        nb_patches = len(dataset)
        print("{} data patches found".format(nb_patches))

        if validation_dataset is not None:
            val_dataloader = DataLoader(validation_dataset, batch_size=validation_batch_size,
                                        shuffle=True, num_workers=4)

            nb_validation = len(validation_dataset)
            print("{} validation patches found".format(nb_validation))

        else:
            print "No validation dataset found... validation skipped"

        if extra_validation_dataset is not None:
            extra_val_dataloader = DataLoader(extra_validation_dataset, batch_size=validation_batch_size,
                                        shuffle=True, num_workers=4)
            extra_nb_validation = len(extra_validation_dataset)
            print("{} extra validation patches found".format(extra_nb_validation))

        print "Start training..."

        for epoch in range(self.start_epoch, self.start_epoch + nb_epochs):
            print "Start training epoch {}".format(epoch)

            self.train(dataloader, epoch, nb_patches, batch_size, show_first_image, supervised, dataset.has_ground_truth())

            if self.killer.kill_now:
                break

            if validation_dataset is not None:
                print "Start validation epoch {}".format(epoch)

                self.validate(val_dataloader, epoch, nb_validation, validation_batch_size, dataset.has_ground_truth())

            if extra_validation_dataset is not None:
                print "Start extra validation epoch {}".format(epoch)

                self.validate(extra_val_dataloader, epoch, extra_nb_validation, validation_batch_size, has_ground_truth=False, extra=True)

            if self.killer.kill_now:
                break

            self.save_checkpoint(epoch, output_file, dataset.has_ground_truth())
            if self.killer.kill_now:
                break

        if self.killer.kill_now:
            print("Graceful exit")
        else:
            print('Finished training')


    def train(self, dataloader, epoch, nb_patches, batch_size=40, show_first_image=False, supervised=False, has_ground_truth=False):
        self.net.train()

        previousTime = time.time()

        loss_function = torch.nn.L1Loss()

        im = numpy.zeros((128, 256), dtype="uint8")

        self.running_loss.append(0)
        self.ground_truth_loss.append(0)
        for i, sample in enumerate(dataloader):

            image, input, ground_truth = sample

            self.optimizer.zero_grad()

            input = Variable(input.float().cuda(), 0)
            image = Variable(image.float().cuda())

            if has_ground_truth:
                ground_truth = Variable(ground_truth.float().cuda())

            output, _, h4pt = self.net(input, image, self.patch_location)

            if has_ground_truth:
                ground_truth_loss = l2_corner_loss(h4pt, ground_truth)
                self.ground_truth_loss[epoch] += ground_truth_loss.data[0] / (nb_patches)

            loss = loss_function(output, input[:, 1])

            if show_first_image:

                im[0:128, 0:128] = input[1, 1, :, :].data.cpu().numpy().astype("uint8")
                im[0:128, 128:256] = output.data[1].cpu().numpy().astype("uint8")
                cv2.imshow("image", im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if supervised:
                ground_truth_loss.backward()
            else:
                loss.backward()

            self.optimizer.step()

            self.running_loss[epoch] += loss.data[0] * batch_size / (nb_patches * 255)

            batch_time = ((time.time() - previousTime) / (60 * (i + 1))) * ((nb_patches / batch_size) - (i + 1))
            sys.stdout.write('\rEpoch {}: training: {}% done, ETA: {} min, AL: {}, CL: {}'.format(
                epoch,
                (i * batch_size * 100) / nb_patches,
                batch_time,
                self.running_loss[epoch] * nb_patches / (batch_size * (i + 1)),
                loss.data[0] / 255))

            if self.killer.kill_now:
                return

        sys.stdout.write('\n')

        currentTime = time.time()
        print('Duration: %.3f seconds' % (currentTime - previousTime))
        previousTime = currentTime

        print('[%d, %5d] loss: %.10f' %
              (epoch, i + 1, self.running_loss[epoch]))
        if has_ground_truth:
            print('[%d, %5d] ground truth loss: %.10f' %
                  (epoch, i + 1, self.ground_truth_loss[epoch]))
        if epoch != 0:
            print('[%d, %5d] loss improvement: %.10f' %
                  (epoch, i + 1, self.running_loss[epoch - 1] - self.running_loss[epoch]))

            if has_ground_truth:
                print('[%d, %5d] ground truth loss improvement: %.10f' %
                      (epoch, i + 1, self.ground_truth_loss[epoch - 1] - self.ground_truth_loss[epoch]))

    def validate(self, val_dataloader, epoch, nb_validation, batch_size=20, has_ground_truth=False, extra=False):

        self.net.eval()

        loss_function = torch.nn.L1Loss()

        self.validation_loss.append(0)
        self.ground_truth_validation_loss.append(0)
        self.extra_validation_loss.append(0)
        for i, sample in enumerate(val_dataloader):

            image, input, ground_truth = sample

            input = Variable(input.float().cuda(), volatile=True)
            image = Variable(image.float().cuda(), volatile=True)

            if has_ground_truth:
                ground_truth = Variable(ground_truth.float().cuda(), volatile=True)

            output, _, h4pt = self.net(input, image, self.patch_location)

            if has_ground_truth:
                ground_truth_loss = l2_corner_loss(h4pt, ground_truth)
                self.ground_truth_validation_loss[epoch] += ground_truth_loss.data[0] / (nb_validation)

            loss = loss_function(output, input[:, 1, :, :])

            if extra:
                self.extra_validation_loss[epoch] += loss.data[0] * batch_size / (nb_validation * 255)
            else:
                self.validation_loss[epoch] += loss.data[0] * batch_size / (nb_validation * 255)

            sys.stdout.write(
                    '\rEpoch {}: validation: {}% done'.format(epoch, (i * batch_size * 100) / nb_validation))

            if self.killer.kill_now:
                return

        sys.stdout.write('\n')

        print('[%d, %5d] validation loss: %.10f' %
              (epoch, i + 1, self.validation_loss[epoch]))
        if has_ground_truth:
            print('[%d, %5d] ground truth loss: %.10f' %
                  (epoch, i + 1, self.ground_truth_validation_loss[epoch]))
        if epoch != 0:
            print('[%d, %5d] validation improvement: %.10f' %
                  (epoch, i + 1, self.validation_loss[epoch - 1] - self.validation_loss[epoch]))
            if has_ground_truth:
                print('[%d, %5d] ground truth validation improvement: %.10f' %
                  (epoch, i + 1, self.ground_truth_validation_loss[epoch - 1] - self.ground_truth_validation_loss[epoch]))


    def evaluate(self, input, image):

        self.net.eval()

        input = Variable(input.float().cuda(), volatile=True)
        image = Variable(image.float().cuda(), volatile=True)

        loss_function = torch.nn.L1Loss()

        time1 = time.time()
        output, h, h4pt = self.net(input, image, self.patch_location)
        torch.cuda.synchronize()

        self.last_evaluation_loss = loss_function(output, input[:, 1])/255

        return output, h, h4pt

    def evaluate_net(self, input):

        self.net.eval()

        time1 = time.time()
        input = Variable(input.float().cuda(), volatile=True)

        h4pt = self.net.forward_net(input)
        torch.cuda.synchronize()


        return h4pt


class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self,signum, frame):

    self.kill_now = True