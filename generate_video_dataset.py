import cv2
import os, csv

dataset_name = "example8"
folder = "video/"
extension = ".mp4"
video_files = ["2"]

if not os.path.exists("datasets/{}".format(dataset_name)):
    os.makedirs("datasets/{}".format(dataset_name))

start_frame = 2940
nb_dataset = 110
frame_distance = 5

crop_size=(256, 256)
patch_size=(128, 128)
patch_location = (128, 128)

for video_file in video_files:

    read_mode = 'wb'
    nb_lines = 0
    if os.path.exists('datasets/{}.csv'.format(dataset_name)):
        read_mode = 'ab'
        nb_lines = sum(1 for line in open('datasets/{}.csv'.format(dataset_name)))
        print "Dataset already exists, appending at # %d" % nb_lines

    cap = cv2.VideoCapture(folder+video_file+extension)

    nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print "{} frames found".format(nb_frames)

    cap.set(1, start_frame)

    ret, color_image1 = cap.read()

    color_image1 = color_image1[240:840,660:1260]
    color_image1 = cv2.resize(color_image1, (256,256), interpolation = cv2.INTER_AREA )
    image1 = cv2.cvtColor(color_image1, cv2.COLOR_BGR2GRAY)

    with open('datasets/{}.csv'.format(dataset_name), read_mode) as csvfile:
        writer = csv.writer(csvfile,delimiter=',')

        for i in range(1, nb_dataset):

            cap.set(1, start_frame+i*frame_distance)

            ret, color_image2 = cap.read()

            color_image2 = color_image2[40:1040, 460:1460]
            color_image2 = cv2.resize(color_image2, (256, 256), interpolation=cv2.INTER_AREA)
            image2 = cv2.cvtColor(color_image2, cv2.COLOR_BGR2GRAY)


            patch1 = image1[patch_location[0] - patch_size[0] / 2:patch_location[0] + patch_size[0] / 2,
                             patch_location[1] - patch_size[1] / 2:patch_location[1] + patch_size[1] / 2]
            patch2 = image2[patch_location[0] - patch_size[0] / 2:patch_location[0] + patch_size[0] / 2,
                     patch_location[1] - patch_size[1] / 2:patch_location[1] + patch_size[1] / 2]

            filenames = ["{}/{}.jpg".format(dataset_name, video_file+str(i+nb_lines)),
                         "{}/{}_1.jpg".format(dataset_name, video_file+str(i+nb_lines)),
                         "{}/{}_2.jpg".format(dataset_name, video_file+str(i+nb_lines)),
                         "{}/{}_3.jpg".format(dataset_name, video_file + str(i + nb_lines))]

            writer.writerow(filenames)

            cv2.imwrite("datasets/{}".format(filenames[0]), image1)
            cv2.imwrite("datasets/{}".format(filenames[1]), patch1)
            cv2.imwrite("datasets/{}".format(filenames[2]), patch2)
            cv2.imwrite("datasets/{}".format(filenames[3]), color_image1)

            color_image1 = color_image2
            image1 = image2
