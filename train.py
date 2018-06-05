from helper.netinstance import NetInstance
from datasets.trainingdataset import TrainingDataset
from datasets.videodataset import VideoDataset
import argparse, os

parser = argparse.ArgumentParser(description='Training a model')

default_batch_size = 40
default_learning_rate = 0.0001
default_training_dataset = "dataset_train_mild.csv"
default_training_dataset_length = 0
default_validation_dataset = "dataset_val_mild.csv"
default_validation_dataset_length = 0
default_number_epochs = 10000

parser.add_argument('-i', '--input', help='input checkpoint')
parser.add_argument('-o', '--output', help='output checkpoint')
parser.add_argument('-td', '--training_dataset', default=default_training_dataset, help='training dataset')
parser.add_argument('-vd', '--validation_dataset', default=default_validation_dataset, help='validation dataset')
parser.add_argument('-evd', '--extra_validation_dataset', default=None, help='extra validation dataset')
parser.add_argument('-tdl', '--training_dataset_length', default=default_training_dataset_length, help='training dataset length', type=int)
parser.add_argument('-vdl', '--validation_dataset_length', default=default_validation_dataset_length, help='validation dataset length', type=int)
parser.add_argument('-b', '--batch', help='batch size', default=default_batch_size, type=int)
parser.add_argument('-lr', '--learning_rate', help='learning rate', default=default_learning_rate, type=float)
parser.add_argument('-ep', '--epochs', help='number of epochs', default=default_number_epochs, type=int)
parser.add_argument('-sr','--safe_reset', help="if output file exists, use that as input file to prevent lost work", action="store_true")
#parser.add_argument('-no_dr','--no_dropout', help="delete dropout", action="store_true")
parser.add_argument('-su','--supervised', help="train supervised", action="store_true")
parser.add_argument('-n','--net', default="original", help="choose net name")
parser.add_argument('-m','--map', default="homographymap", help="choose map name")
parser.add_argument('-t', '--type', default="training", help="type of dataset: training or video")

args = parser.parse_args()
if args.type == "video":
    print "Video dataset"
    dataset = VideoDataset("datasets/%s" % args.training_dataset, args.training_dataset_length)
    val_dataset = VideoDataset("datasets/%s" % args.validation_dataset, args.validation_dataset_length)
else:
    print "Training dataset"
    dataset = TrainingDataset("datasets/%s" % args.training_dataset, args.training_dataset_length)
    val_dataset = TrainingDataset("datasets/%s" % args.validation_dataset, args.validation_dataset_length)

if args.extra_validation_dataset is not None:

    extra_val_dataset = VideoDataset("datasets/%s" % args.extra_validation_dataset)
else:
    extra_val_dataset = None

input_checkpoint = None if args.input is None else "checkpoints/%s" % args.input
output_checkpoint = None if args.output is None else "checkpoints/%s" % args.output

if args.safe_reset and output_checkpoint is not None and os.path.isfile(output_checkpoint):
    print('Safe reset used: output file exists, so use that as input')
    input_checkpoint = output_checkpoint

print "Training parameters: "

print('Input checkpoint: %s' % input_checkpoint)
print('Output checkpoint: %s' % output_checkpoint)
print('Training dataset: %s, size %d' % (args.training_dataset, args.training_dataset_length))
print('Validation dataset: %s, size %d' % (args.validation_dataset, args.validation_dataset_length))
print('Batch size: %d' % args.batch)
print('Learning rate: %f' % args.learning_rate)
print('Number of epochs: %d' % args.epochs)
print('Net name: %s' % args.net)

print "Call trainer with selected parameters..."

netinstance = NetInstance(args.learning_rate, input_file=input_checkpoint, net_name=args.net, map_name=args.map)
netinstance.train_and_validate(dataset, args.batch, validation_dataset=val_dataset, extra_validation_dataset=extra_val_dataset, output_file=output_checkpoint, nb_epochs=args.epochs, supervised=args.supervised)