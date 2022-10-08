SMiLER (Samsung MultiLingual Entity and Relation extraction) corpus was created at Samsung R&D Institute Poland.


This dataset is licensed under the Creative Commons Attribution-Share-Alike 3.0 License (https://creativecommons.org/licenses/by-sa/3.0/).


In total, the corpus contains 1.4 million sentences.

The corpus consists of 32 TSV files. For each of the 14 languages covered, there are 2 files: train and test.



Each TSV file has the following columns (A-G):


A. id


B. entity_1

For English, this is the original entity_1. For other languages, this is entity_1 translated automatically from English.


C. entity_2

For English, this is the original entity_2. For other languages, this is entity_2 translated automatically from English.


D. label

The labels are always in English. They are shared among all the files. The label is the relation between entity_1 and entity_2, e.g. has-child or has-length.


E. text

This is the sentence with annotated entity_1 (<e1>...</e1>) and entity_2 (<e2>...</e2>).

For English, this is the original annotated sentence, copied from the English Wikipedia dump.

For other languages (but not for label "no_relation"), this sentence was copied from the Wikipedia dump in the relevant language, and then annotated automatically.

For "no_relation", this sentence was translated automatically from English and then annotated automatically.

This field is empty whenever the translated entities can't be found in the sentence (such empty fields don't count towards the total number of sentences in the corpus).


F. lang

This is the language of the file.