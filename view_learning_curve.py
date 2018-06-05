import matplotlib.pyplot as plt
from helper.netinstance import NetInstance

checkpoint_filenames = ["checkpoints/realdata-vgg13-double_val.pth.tar"]
labels = ["vgg13"]

# checkpoint_filenames = ["checkpoints/test_extra.pth.tar"]
# labels = ["text_extra"]

settings = {"unsupervised_loss": True, "max_epoch": 60, "start_epoch": 0}
start_extra_validation_epoch = 46


plot_handles = []

for checkpoint_filename,label in zip(checkpoint_filenames, labels):
    print checkpoint_filename
    training_loss, validation_loss, ground_truth_loss, ground_truth_validation_loss, extra_validation_loss = NetInstance(input_file=checkpoint_filename).get_loss()

    print validation_loss
    print extra_validation_loss

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    if settings["unsupervised_loss"]:
        if settings["max_epoch"] > 0:
            training_loss = training_loss[settings["start_epoch"]:settings["max_epoch"]]
            validation_loss = validation_loss[settings["start_epoch"]:settings["max_epoch"]]
            extra_validation_loss = extra_validation_loss[settings["start_epoch"]:settings["max_epoch"]]
        training_plot, = plt.plot(training_loss, label=label)
        validation_plot, = plt.plot(validation_loss, linestyle='dashed', color=training_plot.get_color())
        extra_validation_plot, = plt.plot(range(start_extra_validation_epoch, len(extra_validation_loss)),extra_validation_loss[start_extra_validation_epoch:len(extra_validation_loss)], linestyle='dotted', color=training_plot.get_color())

        plot_handles.append(training_plot)

    else:
        if settings["max_epoch"] > 0:
            ground_truth_loss = ground_truth_loss[0:settings["max_epoch"]]
            ground_truth_validation_loss = ground_truth_validation_loss[0:settings["max_epoch"]]
        training_plot, = plt.plot(ground_truth_loss, label=label)
        validation_plot, = plt.plot(ground_truth_validation_loss, linestyle='dashed', color=training_plot.get_color())
        plt.ylabel("RMSE")
        plot_handles.append(training_plot)

plt.legend(handles=plot_handles)
plt.show()
