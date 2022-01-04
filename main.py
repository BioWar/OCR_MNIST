from argparse import ArgumentParser
import json
import os
import datetime
import string

from dataset_tools.generator import DataGenerator, Tokenizer
from dataset_tools.utils import * 
from Model.model import HTRModel, CustomSchedule, custom_arch_v8
from Model.layers import GatedConv2D, FullGatedConv2D

from PIL import Image
import h5py
from jinja2 import Template
from jiwer import cer 

if __name__=="__main__":
  parser = ArgumentParser(description="MNIST HWR main file")
  parser.add_argument("-c", "--config", required=True, type=str,
                      help="path to config")
  args = parser.parse_args()

  config_path = args.config

  
  with open(config_path, 'r') as f:
      config = json.load(f)

  source = config['source']
  arch = config['arch']
  epochs = config['epochs']
  batch_size = config['batch_size']

  # define paths
  source_path = config['source_path']
  output_path = os.path.join(config['output_path'], source, arch)
  target_path = os.path.join(output_path, "checkpoint_weights.hdf5")
  os.makedirs(output_path, exist_ok=True)

  # define input size, number max of chars per line and list of valid chars
  height, width = config['input_size_hight'], config['input_size_widht']
  input_size = (width, height, 1)
  max_text_length = config['max_text_len']
  charset_base = config['charset_base']

  print("source:", source_path)
  print("output", output_path)
  print("target", target_path)
  print("charset:", charset_base)

  dataset = h5py.File("dataset.hdf5", "r")

  dtgen = DataGenerator(source="dataset.hdf5",
                        batch_size=batch_size,
                        charset=charset_base,
                        max_text_length=max_text_length)

  print(f"Train images: {dtgen.size['train']}")
  print(f"Validation images: {dtgen.size['valid']}")
  print(f"Test images: {dtgen.size['test']}")

  # create and compile HTRModel
  model = HTRModel(architecture=arch,
                   input_size=input_size,
                   vocab_size=dtgen.tokenizer.vocab_size,
                   beam_width=10,
                   stop_tolerance=20,
                   reduce_tolerance=15)

  model.compile(learning_rate=0.001)
  # model.summary(output_path, "summary.txt")
  model.summary()

  # get default callbacks and load checkpoint weights file (HDF5) if exists
  model.load_checkpoint(target=target_path)

  print(config["train"])
  if config["train"]:
    print("Training started...")
    callbacks = model.get_callbacks(logdir=output_path, checkpoint=target_path, verbose=1)

    # to calculate total and average time per epoch
    start_time = datetime.datetime.now()

    h = model.fit(x=dtgen.next_train_batch(),
                  epochs=epochs,
                  steps_per_epoch=dtgen.steps['train'],
                  validation_data=dtgen.next_valid_batch(),
                  validation_steps=dtgen.steps['valid'],
                  callbacks=callbacks,
                  shuffle=True,
                  verbose=1)

    total_time = datetime.datetime.now() - start_time

    loss = h.history['loss']
    val_loss = h.history['val_loss']

    min_val_loss = min(val_loss)
    min_val_loss_i = val_loss.index(min_val_loss)

    time_epoch = (total_time / len(loss))
    total_item = (dtgen.size['train'] + dtgen.size['valid'])

    t_corpus = "\n".join([
        f"Total train images:      {dtgen.size['train']}",
        f"Total validation images: {dtgen.size['valid']}",
        f"Batch:                   {dtgen.batch_size}\n",
        f"Total time:              {total_time}",
        f"Time per epoch:          {time_epoch}",
        f"Time per item:           {time_epoch / total_item}\n",
        f"Total epochs:            {len(loss)}",
        f"Best epoch               {min_val_loss_i + 1}\n",
        f"Training loss:           {loss[min_val_loss_i]:.8f}",
        f"Validation loss:         {min_val_loss:.8f}"
    ])

    with open(os.path.join(output_path, "train.txt"), "w") as lg:
        lg.write(t_corpus)
        print(t_corpus)
  else:
    print("Evaluation started...")
    start_time = datetime.datetime.now()

    # predict() function will return the predicts with the probabilities
    predicts, _ = model.predict(x=dtgen.next_test_batch(),
                                steps=dtgen.steps['test'],
                                ctc_decode=True,
                                verbose=1)

    # decode to string
    predicts = [dtgen.tokenizer.decode(x[0]) for x in predicts]
    ground_truth_tmp = [re.sub(' +', ' ', x.decode().lstrip().rstrip()) for x in dtgen.dataset['test']['gt']]
    ground_truth = [x.decode() for x in dtgen.dataset['test']['gt']]

    total_time = datetime.datetime.now() - start_time

    items = []

    # mount predict corpus file
    with open(os.path.join(output_path, "predict.txt"), "w") as lg:
        for pd, gt in zip(predicts, ground_truth):
            lg.write(f"TE_L {gt}\nTE_P {pd}\n")
       
    os.makedirs(os.path.join(output_path, "tmp"), exist_ok=True)
    correctly_pred = 0
    all_gt = 0
    for i, item in enumerate(dtgen.dataset['test']['dt'][:]):
        print("=" * 30, "\n")
        img = adjust_to_see(item)
        im = Image.fromarray(img)
        # img_path = os.path.join(output_path, "tmp", f"{i}.jpeg")
        img_path = f"tmp/{i}.jpeg"
        im.save(img_path)
        print(ground_truth[i])
        print(predicts[i], "\n")
        correct = int(predicts[i] == ground_truth_tmp[i])
        correctly_pred += correct
        all_gt += 1
        items.append(dict({"id": i, "image": img_path, "gt": ground_truth[i], "pred": predicts[i], "correct": correct}))

    with open(os.path.join(output_path, "train.txt")) as f:
      train_lines = f.readlines()
    
    train_stats = []
    for line in train_lines:
      value = {"text": line}
      print(line)
      train_stats.append(value)

    cer = cer(ground_truth_tmp, predicts)
    train_stats.append({"text": f"Test CER: {cer}"})
    train_stats.append({"text": f"Correctly predicted vs. All = {correctly_pred}/{all_gt} = {np.round(correctly_pred/all_gt, 4)*100}%"})
    
    for i in range(20):
      print(f"|{ground_truth_tmp[i]}| - |{predicts[i]}|")

    with open(config["template_path"]) as file_:
      template = Template(file_.read())

    html_content = template.render(items=items, train_stats=train_stats)
    with open(os.path.join(output_path, "results_evaluation.html"), "w") as text_file:
      text_file.write(html_content)

  print("End.")