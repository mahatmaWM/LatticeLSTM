# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-14 17:34:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-01-29 15:26:51

from .functions import *
from .gazetteer import Gazetteer
import logging

START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"
NULLKEY = "-null-"


class Data:
    def __init__(self):
        self.MAX_SENTENCE_LENGTH = 50
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = True
        self.norm_word_emb = True
        self.norm_biword_emb = True
        self.norm_gaz_emb = False

        self.word_alphabet = Alphabet('word')
        self.biword_alphabet = Alphabet('biword')
        self.char_alphabet = Alphabet('character')

        # self.word_alphabet.add(START)
        # self.word_alphabet.add(UNKNOWN)
        # self.char_alphabet.add(START)
        # self.char_alphabet.add(UNKNOWN)
        # self.char_alphabet.add(PADDING)

        self.label_alphabet = Alphabet('label', True)
        self.gaz_lower = False
        self.gaz = Gazetteer(self.gaz_lower)
        self.gaz_alphabet = Alphabet('gaz')
        self.HP_fix_gaz_emb = False
        self.HP_use_gaz = True

        self.tagScheme = "NoSeg"
        self.char_features = "LSTM"

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []
        self.use_bigram = True
        self.word_emb_dim = 50
        self.biword_emb_dim = 50
        self.char_emb_dim = 30
        self.gaz_emb_dim = 50
        self.gaz_dropout = 0.5
        self.pretrain_word_embedding = None
        self.pretrain_biword_embedding = None
        self.pretrain_gaz_embedding = None
        self.label_size = 0
        self.word_alphabet_size = 0
        self.biword_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0

        # hyperparameters
        self.HP_iteration = 10
        self.HP_batch_size = 10
        self.HP_char_hidden_dim = 50
        self.HP_hidden_dim = 200
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True
        self.HP_use_char = False
        self.HP_gpu = False
        self.HP_lr = 0.015
        self.HP_lr_decay = 0.05
        self.HP_clip = 5.0
        self.HP_momentum = 0

    def show_data_summary(self):
        logging.info("DATA SUMMARY START:")
        logging.info("     Tag          scheme: %s" % (self.tagScheme))
        logging.info("     MAX SENTENCE LENGTH: %s" % (self.MAX_SENTENCE_LENGTH))
        logging.info("     MAX   WORD   LENGTH: %s" % (self.MAX_WORD_LENGTH))
        logging.info("     Number   normalized: %s" % (self.number_normalized))
        logging.info("     Use          bigram: %s" % (self.use_bigram))
        logging.info("     Word  alphabet size: %s" % (self.word_alphabet_size))
        logging.info("     Biword alphabet size: %s" % (self.biword_alphabet_size))
        logging.info("     Char  alphabet size: %s" % (self.char_alphabet_size))
        logging.info("     Gaz   alphabet size: %s" % (self.gaz_alphabet.size()))
        logging.info("     Label alphabet size: %s" % (self.label_alphabet_size))
        logging.info("     Word embedding size: %s" % (self.word_emb_dim))
        logging.info("     Biword embedding size: %s" % (self.biword_emb_dim))
        logging.info("     Char embedding size: %s" % (self.char_emb_dim))
        logging.info("     Gaz embedding size: %s" % (self.gaz_emb_dim))
        logging.info("     Norm     word   emb: %s" % (self.norm_word_emb))
        logging.info("     Norm     biword emb: %s" % (self.norm_biword_emb))
        logging.info("     Norm     gaz    emb: %s" % (self.norm_gaz_emb))
        logging.info("     Norm   gaz  dropout: %s" % (self.gaz_dropout))
        logging.info("     Train instance number: %s" % (len(self.train_texts)))
        logging.info("     Dev   instance number: %s" % (len(self.dev_texts)))
        logging.info("     Test  instance number: %s" % (len(self.test_texts)))
        logging.info("     Raw   instance number: %s" % (len(self.raw_texts)))
        logging.info("     Hyperpara  iteration: %s" % (self.HP_iteration))
        logging.info("     Hyperpara  batch size: %s" % (self.HP_batch_size))
        logging.info("     Hyperpara          lr: %s" % (self.HP_lr))
        logging.info("     Hyperpara    lr_decay: %s" % (self.HP_lr_decay))
        logging.info("     Hyperpara     HP_clip: %s" % (self.HP_clip))
        logging.info("     Hyperpara    momentum: %s" % (self.HP_momentum))
        logging.info("     Hyperpara  hidden_dim: %s" % (self.HP_hidden_dim))
        logging.info("     Hyperpara     dropout: %s" % (self.HP_dropout))
        logging.info("     Hyperpara  lstm_layer: %s" % (self.HP_lstm_layer))
        logging.info("     Hyperpara      bilstm: %s" % (self.HP_bilstm))
        logging.info("     Hyperpara         GPU: %s" % (self.HP_gpu))
        logging.info("     Hyperpara     use_gaz: %s" % (self.HP_use_gaz))
        logging.info("     Hyperpara fix gaz emb: %s" % (self.HP_fix_gaz_emb))
        logging.info("     Hyperpara    use_char: %s" % (self.HP_use_char))
        if self.HP_use_char:
            logging.info("             Char_features: %s" % (self.char_features))
        logging.info("DATA SUMMARY END.")
        sys.stdout.flush()

    def refresh_label_alphabet(self, input_file):
        old_size = self.label_alphabet_size
        self.label_alphabet.clear(True)
        in_lines = open(input_file, 'r').readlines()
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split()
                label = pairs[-1]
                self.label_alphabet.add(label)
        self.label_alphabet_size = self.label_alphabet.size()
        startS = False
        startB = False
        for label, _ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"
        self.fix_alphabet()
        logging.info("Refresh label alphabet finished: old:%s -> new:%s" % (old_size, self.label_alphabet_size))

    def build_alphabet(self, input_file):
        in_lines = open(input_file, 'r').readlines()
        for idx in range(len(in_lines)):
            line = in_lines[idx]
            if len(line) > 2:
                pairs = line.strip().split()
                word = pairs[0]
                if self.number_normalized:
                    word = normalize_word(word)
                label = pairs[-1]
                self.label_alphabet.add(label)
                self.word_alphabet.add(word)
                if idx < len(in_lines) - 1 and len(in_lines[idx + 1]) > 2:
                    biword = word + in_lines[idx + 1].strip().split()[0]
                else:
                    biword = word + NULLKEY
                self.biword_alphabet.add(biword)
                for char in word:
                    self.char_alphabet.add(char)
        self.word_alphabet_size = self.word_alphabet.size()
        self.biword_alphabet_size = self.biword_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()
        startS = False
        startB = False
        for label, _ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"

    def build_gaz_file(self, gaz_file):
        # build gaz file,initial read gaz embedding file
        if gaz_file:
            fins = open(gaz_file, 'r').readlines()
            for fin in fins:
                fin = fin.strip().split()[0]
                if fin:
                    self.gaz.insert(fin, "one_source")
            logging.info("Load gaz file: %s, total size: %s", gaz_file, self.gaz.size())
        else:
            logging.info("Gaz file is None, load nothing")

    def build_gaz_alphabet(self, input_file):
        in_lines = open(input_file, 'r').readlines()
        word_list = []
        for line in in_lines:
            if len(line) > 3:
                word = line.split()[0]
                if self.number_normalized:
                    word = normalize_word(word)
                word_list.append(word)
            else:
                w_length = len(word_list)
                for idx in range(w_length):
                    matched_entity = self.gaz.enumerateMatchList(word_list[idx:])
                    for entity in matched_entity:
                        # logging.info entity, self.gaz.searchId(entity),self.gaz.searchType(entity)
                        self.gaz_alphabet.add(entity)
                word_list = []
        logging.info("gaz alphabet size: %s", self.gaz_alphabet.size())

    def fix_alphabet(self):
        self.word_alphabet.close()
        self.biword_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close()
        self.gaz_alphabet.close()

    def build_word_pretrain_emb(self, emb_path):
        logging.info("build word pretrain emb...")
        self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(emb_path, self.word_alphabet,
                                                                                   self.word_emb_dim,
                                                                                   self.norm_word_emb)

    def build_biword_pretrain_emb(self, emb_path):
        logging.info("build biword pretrain emb...")
        self.pretrain_biword_embedding, self.biword_emb_dim = build_pretrain_embedding(emb_path, self.biword_alphabet,
                                                                                       self.biword_emb_dim,
                                                                                       self.norm_biword_emb)

    def build_gaz_pretrain_emb(self, emb_path):
        logging.info("build gaz pretrain emb...")
        self.pretrain_gaz_embedding, self.gaz_emb_dim = build_pretrain_embedding(emb_path, self.gaz_alphabet,
                                                                                 self.gaz_emb_dim, self.norm_gaz_emb)

    def generate_instance(self, input_file, name):
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_seg_instance(input_file, self.word_alphabet, self.biword_alphabet,
                                                                 self.char_alphabet, self.label_alphabet,
                                                                 self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_seg_instance(input_file, self.word_alphabet, self.biword_alphabet,
                                                             self.char_alphabet, self.label_alphabet,
                                                             self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_seg_instance(input_file, self.word_alphabet, self.biword_alphabet,
                                                               self.char_alphabet, self.label_alphabet,
                                                               self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_seg_instance(input_file, self.word_alphabet, self.biword_alphabet,
                                                             self.char_alphabet, self.label_alphabet,
                                                             self.number_normalized, self.MAX_SENTENCE_LENGTH)
        else:
            logging.info("Error: you can only generate train/dev/test instance! Illegal input: %s" % (name))

    def generate_instance_with_gaz(self, input_file, name):
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_instance_with_gaz(input_file, self.gaz, self.word_alphabet,
                                                                      self.biword_alphabet, self.char_alphabet,
                                                                      self.gaz_alphabet, self.label_alphabet,
                                                                      self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance_with_gaz(input_file, self.gaz, self.word_alphabet,
                                                                  self.biword_alphabet, self.char_alphabet,
                                                                  self.gaz_alphabet, self.label_alphabet,
                                                                  self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance_with_gaz(input_file, self.gaz, self.word_alphabet,
                                                                    self.biword_alphabet, self.char_alphabet,
                                                                    self.gaz_alphabet, self.label_alphabet,
                                                                    self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_instance_with_gaz(input_file, self.gaz, self.word_alphabet,
                                                                  self.biword_alphabet, self.char_alphabet,
                                                                  self.gaz_alphabet, self.label_alphabet,
                                                                  self.number_normalized, self.MAX_SENTENCE_LENGTH)
        else:
            logging.info("Error: you can only generate train/dev/test instance! Illegal input: %s" % (name))

    def write_decoded_results(self, output_file, predict_results, name):
        fout = open(output_file, 'w')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
            content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            logging.info(
                "Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert (sent_num == len(content_list))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx])
            for idy in range(sent_length):
                # content_list[idx] is a list with [word, char, label]
                fout.write(content_list[idx][0][idy].encode('utf-8') + " " + predict_results[idx][idy] + '\n')
            fout.write('\n')
        fout.close()
        logging.info("Predict %s result has been written into file. %s" % (name, output_file))
