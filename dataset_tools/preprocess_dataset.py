import h5py
import multiprocessing
import xml.etree.ElementTree as ET

from glob import glob
from tqdm import tqdm
from functools import partial

from utils import *

from argparse import ArgumentParser

class Dataset():
    """Dataset class to read images and sentences from base (raw files)"""

    def __init__(self, source, transcription):
        self.source = source
        self.transcription = transcription
        self.dataset = None
        self.partitions = ['train', 'valid', 'test']

    def read_partitions(self):
        """Read images and sentences from dataset"""

        dataset = self.mnist_hwr()

        if not self.dataset:
            self.dataset = self._init_dataset()

        for y in self.partitions:
            self.dataset[y]['dt'] += dataset[y]['dt']
            self.dataset[y]['gt'] += dataset[y]['gt']

    def save_partitions(self, target, image_input_size, max_text_length):
        """Save images and sentences from dataset"""

        os.makedirs(os.path.dirname(target), exist_ok=True)
        total = 0

        with h5py.File(target, "w") as hf:
            for pt in self.partitions:
                self.dataset[pt] = self.check_text(self.dataset[pt], max_text_length)
                size = (len(self.dataset[pt]['dt']),) + image_input_size[:2]
                total += size[0]

                dummy_image = np.zeros(size, dtype=np.uint8)
                dummy_sentence = [("c" * max_text_length).encode()] * size[0]

                hf.create_dataset(f"{pt}/dt", data=dummy_image, compression="gzip", compression_opts=9)
                hf.create_dataset(f"{pt}/gt", data=dummy_sentence, compression="gzip", compression_opts=9)

        pbar = tqdm(total=total)
        batch_size = 1024

        for pt in self.partitions:
            for batch in range(0, len(self.dataset[pt]['gt']), batch_size):
                images = []

                with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                    r = pool.map(partial(preprocess, input_size=image_input_size),
                                 self.dataset[pt]['dt'][batch:batch + batch_size])
                    images.append(r)
                    pool.close()
                    pool.join()

                with h5py.File(target, "a") as hf:
                    hf[f"{pt}/dt"][batch:batch + batch_size] = images
                    hf[f"{pt}/gt"][batch:batch + batch_size] = [s.encode() for s in self.dataset[pt]
                                                                ['gt'][batch:batch + batch_size]]
                    pbar.update(batch_size)

    def _init_dataset(self):
        dataset = dict()

        for i in self.partitions:
            dataset[i] = {"dt": [], "gt": []}

        return dataset

    def _shuffle(self, *ls):
        random.seed(42)

        if len(ls) == 1:
            li = list(*ls)
            random.shuffle(li)
            return li

        li = list(zip(*ls))
        random.shuffle(li)
        return zip(*li)

    def mnist_hwr(self):
        """MNIST HWR dataset reader"""
        lines = open(self.transcription).read().splitlines()[1:]
        gt_dict = dict()

        all_files = []
        for line in lines:
            split = line.split(',')
            split[1] = split[1].replace("-", "").replace("|", " ")
            split[1] = split[1].replace("s_", "")
            gt_dict[split[0]] = split[1]
            all_files.append(split[0])

        img_path = os.path.join(self.source)
        dataset = self._init_dataset()
        n = len(all_files)
        train, valid, test = all_files[:int(0.90*n)], all_files[int(0.90*n):int(0.95*n)], all_files[int(0.95*n):]
        paths = {"train":train, "valid":valid, "test":test}

        partitions = ['train', 'valid', 'test']
        for i in partitions:
            for line in paths[i]:
                dataset[i]['dt'].append(os.path.join(img_path, f"{line}.jpeg"))
                dataset[i]['gt'].append(gt_dict[line])
        return dataset

    @staticmethod
    def check_text(data, max_text_length=128):
        """Checks if the text has more characters instead of punctuation marks"""

        dt = {'gt': list(data['gt']), 'dt': list(data['dt'])}

        for i in reversed(range(len(dt['gt']))):
            text = text_standardize(dt['gt'][i])
            strip_punc = text.strip(string.punctuation).strip()
            no_punc = text.translate(str.maketrans("", "", string.punctuation)).strip()

            length_valid = (len(text) > 1) and (len(text) < max_text_length)
            text_valid = (len(strip_punc) > 1) and (len(no_punc) > 1)

            if (not length_valid) or (not text_valid):
                dt['gt'].pop(i)
                dt['dt'].pop(i)
                continue

        return dt

if __name__=="__main__":
    parser = ArgumentParser(description="MNIST HWR dataset generation")
    parser.add_argument("-i", "--input", required=True, type=str,
                      help="path to directory where images stored")
    parser.add_argument("-t", "--transcription", required=True, type=str,
                      help="path to transcription file")
    parser.add_argument("-o", "--output", required=True, type=str,
                      help="path to directory where dataset HDF5 file will be saved")
    args = parser.parse_args()

    input_path = args.input
    transcription_path = args.transcription
    output_path = args.output

    print(f"Path to images: {input_path}")
    print(f"Path to transcription file: {transcription_path}")
    print(f"Location of created file: {output_path}")

    print("Creating dataset...")
    dataset = Dataset(source=input_path, transcription=transcription_path)

    input_size = (280, 28, 1)
    max_text_length = 16
    charset_base = "0123456789 "

    print(f"Size of images: {input_size}")
    print(f"Max text length: {max_text_length}")
    print(f"Charset: '{charset_base}'")

    print("Reading partitions...")
    print("90% - training; 5% - validation; 5% - test")
    dataset.read_partitions()
    print("Saving HDF5 file...")
    dataset.save_partitions(output_path, input_size, max_text_length)
    print("Completed.")