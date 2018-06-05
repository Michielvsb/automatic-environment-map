from model.homographymap import HomographyMap
import cv2, os, torch, csv, numpy, sys
from random import randint
from torch.autograd import Variable

batch_size = 600
nb_patches = 5000
image_size = (256,256)
patch_size = (128,128)
patch_location =(128,128)
directory = "../UnsupervisedHomographyNet/test2017"
amount = 8
translation = 8
data_filename="test_mild"


nb_batches = nb_patches/batch_size
nb_data = len(os.listdir(directory))
nb_batches_dataset = nb_data/batch_size

print "Number of data files: {} images".format(nb_data)

width, height = patch_size
center_x,center_y = patch_location
p1, p2, p3, p4 = [center_x-width/2,center_y-height/2], [center_x-width/2-1, center_y+height/2], [center_x+width/2, center_y+height/2], [center_x+width/2, center_y-height/2]


homographymap = HomographyMap(patch_size, patch_location)
homographymap.cuda()
homographymap.eval()

if not os.path.exists("datasets/{}".format(data_filename)):
    os.makedirs("datasets/{}".format(data_filename))

def prepare_image(image, image_size):
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, image_size)
    return image

def generate_random_h4pt(amount):
     h4pt = torch.cuda.FloatTensor([[randint(-amount, amount), randint(-amount, amount)], \
                             [randint(-amount, amount), randint(-amount, amount)], \
                             [randint(-amount, amount), randint(-amount, amount)], \
                             [randint(-amount, amount), randint(-amount, amount)]]
                                   )
     translate_w = randint(-translation, translation)
     translate_h = randint(-translation, translation)
     h4pt += torch.cuda.FloatTensor([[translate_w, translate_h], \
                             [translate_w, translate_h], \
                             [translate_w, translate_h], \
                             [translate_w, translate_h]]
                                   )
     return h4pt

batch_cnt = 0
j = 0
images = numpy.zeros((batch_size, image_size[0], image_size[1]), dtype="uint8")
with open('datasets/{}.csv'.format(data_filename), 'wb') as csvfile:
    writer = csv.writer(csvfile,delimiter=',')

    while True:

        for filename in os.listdir(directory):
            if j%(batch_size+1) == batch_size:

                h4pt = Variable(torch.stack([generate_random_h4pt(amount) for i in range(batch_size)]), volatile=True)

                _, patch1 = homographymap(Variable(torch.zeros(batch_size, 4, 2), volatile=True).cuda(),
                                             Variable(torch.from_numpy(images), volatile=True).cuda())
                h, patch2 = homographymap(h4pt, Variable(torch.from_numpy(images), volatile=True).cuda())

                patch1 = patch1.cpu()
                patch2 = patch2.cpu()

                h4pt = h4pt.cpu().data.numpy()

                for i in range(batch_size):
                    filenames = ["{}/patch{}.jpg".format(data_filename, j-batch_size+i), "{}/patch{}_1.jpg".format(data_filename, j-batch_size+i),
                                 "{}/patch{}_2.jpg".format(data_filename, j-batch_size+i)]


                    writer.writerow(filenames + h4pt[i].reshape(8).tolist())

                    cv2.imwrite("datasets/{}".format(filenames[0]), images[i])
                    cv2.imwrite("datasets/{}".format(filenames[1]), patch1.data[i].numpy().astype("uint8"))
                    cv2.imwrite("datasets/{}".format(filenames[2]), patch2.data[i].numpy().astype("uint8"))

                batch_cnt +=1
                sys.stdout.write('\rImages done: {}/{} images, {}%'.format(j,nb_patches,j*100/nb_patches))

                if batch_cnt%nb_batches_dataset == 0:
                    break

            images[j%batch_size] = prepare_image(directory+"/"+filename, image_size)

            j+=1

            if batch_cnt >= nb_batches:
                break
        if batch_cnt >= nb_batches:
            break

sys.stdout.write('\n')
print "Number of patches: {}".format(j)


