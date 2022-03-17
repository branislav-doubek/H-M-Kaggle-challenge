import collections
import json
import os
import random
import re

from absl import app
from absl import flags
from absl import logging
import pickle
import pandas as pd
import tensorflow as tf

FLAGS = flags.FLAGS

CSV_DIR = "drive/MyDrive/csv_data"
TRAIN_FILENAME = "transactions_train.csv"
ARTICLE_FILENAME = "articles.csv"
CUSTOMER_FILENAME = "customers.csv"
ARTICLE_DATA_COLUMNS = ['article_id', 'product_code', 'product_type_no',
                      'graphical_appearance_no', 'colour_group_code',
                      'perceived_colour_value_id', 'perceived_colour_master_id', 
                      'department_no', 'index_code', 
                      'index_group_no', 'section_no', 'garment_group_no']
OUTPUT_TRAINING_DATA_FILENAME = "train_recommenders_v1.tfrecord"
OUTPUT_TESTING_DATA_FILENAME = "test_recommenders_v1.tfrecord"
STRING_CUSTOMER_COLS = ["club_member_status", "fashion_news_frequency"]
STRING_ARTICLE_COLS = ['index_code']

CATEG_CUSTOMER_COLS = ['club_member_status', 'fashion_news_frequency',
                        'postal_code', 'FN', 'Active', 'sales_channel_id'
                      ]
CATEG_ARTICLE_COLS = ['product_code', 'product_type_no',
                      'graphical_appearance_no', 'colour_group_code',
                      'perceived_colour_value_id', 'perceived_colour_master_id', 
                      'department_no', 'index_code', 
                      'index_group_no', 'section_no', 'garment_group_no']


def define_flags():
  """Define flags."""
  flags.DEFINE_string("data_dir", "/tmp",
                      "Path to download and store movielens data.")
  flags.DEFINE_string("output_dir", 'drive/MyDrive/output_hm',
                      "Path to the direcNonetory of output files.")
  flags.DEFINE_bool("build_vocabs", True,
                    "If yes, generate articles feature vocabs.")
  flags.DEFINE_integer("min_timeline_length", 2,
                       "The minimum timeline length to construct examples.")
  flags.DEFINE_integer("max_context_length", 5,
                       "The maximum length of user context history.")
  flags.DEFINE_integer("max_context_movie_genre_length", 10,
                       "The maximum length of user context history.")
  flags.DEFINE_integer(
      "min_rating", None, "Minimum rating of movie that will be used to in "
      "training data")
  flags.DEFINE_float("train_data_fraction", 0.9, "Fraction of training data.")


class ArticleInfo(
    collections.namedtuple(
        "ArticleInfo", [
                      "article_id", 
                      "product_code", 
                      "product_type_no",
                      "graphical_appearance_no", 
                      "colour_group_code", 
                      "perceived_colour_value_id", 
                      "perceived_colour_master_id",
                      "department_no", 
                      "index_code", 
                      "index_group_no",
                      "section_no", 
                      "garment_group_no",
                      "timestamp",
                      "price",
                      "sales_channel_id",
                      "FN",
                      "Active",
                      "club_member_status",
                      "fashion_news_frequency",
                      "age",
                      "postal_code"
                    ])):
  """Data holder of basic information of an article."""
  __slots__ = ()

  def __new__(cls,
              article_id = 0, 
              product_code = 0, 
              product_type_no = 0,
              graphical_appearance_no = 0, 
              colour_group_code = 0, 
              perceived_colour_value_id = 0, 
              perceived_colour_master_id = 0,
              department_no = 0, 
              index_code = "UNK", 
              index_group_no = 0,
              section_no = 0, 
              garment_group_no = 0,
              timestamp = 0,
              price = 0,
              sales_channel_id = 0,
              FN = 0,
              Active = 0,
              club_member_status = "UNK",
              fashion_news_frequency = "UNK",
              age=0,
              postal_code=""):
    return super(ArticleInfo, cls)\
            .__new__(cls, article_id, product_code, 
                     product_type_no,
                     graphical_appearance_no, 
                     colour_group_code, 
                     perceived_colour_value_id,
                     perceived_colour_master_id,
                     department_no, index_code,
                     index_group_no, section_no,
                     garment_group_no, timestamp,
                     price, sales_channel_id,
                     FN, Active, club_member_status,
                     fashion_news_frequency, age,
                     postal_code)


def convert_to_timelines(dataset):
  """Convert ratings data to user."""
  timelines = collections.defaultdict(list)
  article_counts = collections.Counter()
  print(dataset.columns)
  for t_dat, customer_id, article_id, price, sales_channel_id, \
      timestamp, fn, active, club_member_status, fashion_news_frequency, \
      age, postal_code  in dataset.values:
    timelines[customer_id].append(
      ArticleInfo(article_id=article_id,
                  price=price,
                  sales_channel_id=sales_channel_id, 
                  FN=fn, 
                  Active=active, 
                  club_member_status=club_member_status, 
                  fashion_news_frequency=fashion_news_frequency,
                  age=age,
                  postal_code=postal_code,
                  timestamp=timestamp
                  )
    )
    article_counts[article_id] += 1
  # Sort per-user timeline by timestamp
  for (customer_id, context) in timelines.items():
    context.sort(key=lambda x: x.timestamp)
    timelines[customer_id] = context
  return timelines, article_counts

def read_data(data_directory):
  """Read h&m data into dataframe."""
  train_ds = pd.read_csv(
      os.path.join(data_directory, TRAIN_FILENAME),
      dtype={'article_id': int},
      parse_dates=['t_dat']
  )
  train_ds['timestamp'] = train_ds['t_dat'].astype(int)
  article_ds = pd.read_csv(
      os.path.join(data_directory, ARTICLE_FILENAME),
      dtype={'article_id': int}
  )[ARTICLE_DATA_COLUMNS]
  customer_ds = pd.read_csv(
    os.path.join(data_directory, CUSTOMER_FILENAME),
    dtype={'club_member_status': str, 'fashion_news_frequency': str}
  )
  merged_train = train_ds.merge(customer_ds, on='customer_id')
  merged_train[STRING_CUSTOMER_COLS] = merged_train[STRING_CUSTOMER_COLS].fillna('UNK')
  article_ds[STRING_ARTICLE_COLS] = article_ds[STRING_ARTICLE_COLS].fillna('UNK')
  return merged_train.sort_values('timestamp').fillna(0).tail(2500000), article_ds.fillna(0)

def generate_articles_dict(articles_df):
  print(articles_df.columns)
  """Generates article dictionary from h&m dataframe."""
  articles_dict = {article_id: ArticleInfo(article_id=article_id,
                                           product_code=product_code,
                                           product_type_no=product_type_no,
                                           graphical_appearance_no=graphical_appearance_no,
                                           colour_group_code=colour_group_code,
                                           perceived_colour_value_id=perceived_colour_value_id, 
                                           perceived_colour_master_id=perceived_colour_master_id,
                                           department_no=department_no,
                                           index_code=index_code,
                                           index_group_no=index_group_no,
                                           section_no=section_no,
                                           garment_group_no=garment_group_no
                                           )
                              for article_id, product_code, \
                                  product_type_no, graphical_appearance_no, \
                                  colour_group_code, \
                                  perceived_colour_value_id, \
                                  perceived_colour_master_id, \
                                  department_no, index_code, \
                                  index_group_no, section_no, \
                                  garment_group_no in articles_df[ARTICLE_DATA_COLUMNS].values
                  }
  articles_dict[0] = ArticleInfo()
  return articles_dict


def generate_examples_from_single_timeline(timeline,
                                           articles_dict,
                                           max_context_len=100,):
  """Generate TF examples from a single user timeline.
  Generate TF examples from a single user timeline. Timeline with length less
  than minimum timeline length will be skipped. And if context user history
  length is shorter than max_context_len, features will be padded with default
  values.
  Args:
    timeline: The timeline to generate TF examples from.
    articles_dict: Dictionary of all ArticleInfos.
    max_context_len: The maximum length of the context. If the context history
      length is less than max_context_length, features will be padded with
      default values.
  Returns:
    examples: Generated examples from this single timeline.
  """
  examples = []
  for label_idx in range(1, len(timeline)):
    start_idx = max(0, label_idx - max_context_len)
    context = timeline[start_idx:label_idx]
    # Pad context with out-of-vocab article id 0.
    while len(context) < max_context_len:
      context.append(ArticleInfo())
    label = timeline[label_idx]
    label_article_id = int(label.article_id)
    label_product_code = int(articles_dict[label.article_id].product_code)
    label_product_type_no = int(articles_dict[label.article_id].product_type_no)
    label_graphical_appearance_no = int(articles_dict[label.article_id].graphical_appearance_no)
    label_colour_group_code = int(articles_dict[label.article_id].colour_group_code)
    label_perceived_colour_value_id = int(articles_dict[label.article_id].perceived_colour_value_id)
    label_perceived_colour_master_id = int(articles_dict[labelarticle.article_id].perceived_colour_master_id)
    label_department_no = int(articles_dict[label.article_id].department_no)
    label_index_code = tf.compat.as_bytes(articles_dict[label.article_id].index_code)
    label_index_group_no = int(articles_dict[label.article_id].index_group_no)
    label_section_no = int(articles_dict[label.article_id].section_no)
    label_garment_group_no = int(articles_dict[label.article_id].garment_group_no)
    
    context_article_id = [int(article.article_id) for article in context]
    context_product_code = [int(articles_dict[article.article_id].product_code) for article in context]
    context_product_type_no = [int(articles_dict[article.article_id].product_type_no) for article in context]
    context_graphical_appearance_no = [int(articles_dict[article.article_id].graphical_appearance_no) for article in context]
    context_colour_group_code = [int(articles_dict[article.article_id].colour_group_code) for article in context]
    context_perceived_colour_value_id = [int(articles_dict[article.article_id].perceived_colour_value_id) for article in context]
    context_perceived_colour_master_id = [int(articles_dict[article.article_id].perceived_colour_master_id) for article in context]
    context_department_no = [int(articles_dict[article.article_id].department_no) for article in context]
    context_index_code = [tf.compat.as_bytes(articles_dict[article.article_id].index_code) for article in context]
    context_index_group_no = [int(articles_dict[article.article_id].index_group_no) for article in context]
    context_section_no = [int(articles_dict[article.article_id].section_no) for article in context]
    context_garment_group_no = [int(articles_dict[article.article_id].garment_group_no) for article in context]
    context_timestamp = [int(article.timestamp) for article in context]
    context_price = [float(article.price) for article in context]
    context_sales_channel_id = [int(article.sales_channel_id) for article in context]
    context_fn = [int(article.FN) for article in context[:1]]
    context_active = [int(article.Active) for article in context[:1]]
    context_club_member_status = [tf.compat.as_bytes(article.club_member_status) for article in context[:1]]
    context_fashion_news_frequency = [tf.compat.as_bytes(article.fashion_news_frequency) for article in context[:1]]
    context_age = [int(article.age) for article in context[:1]]
    context_postal_code = [tf.compat.as_bytes(article.postal_code) for article in context[:1]]
    feature = {
        "context_article_id":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=context_article_id)),
        "context_product_code":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=context_product_code)),
        "context_product_type_no":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=context_product_type_no)),
        "context_graphical_appearance_no":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=context_graphical_appearance_no)),
        "context_colour_group_code":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=context_colour_group_code)),
        "context_perceived_colour_value_id":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=context_perceived_colour_value_id)),
        "context_perceived_colour_master_id":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=context_perceived_colour_master_id)),
        "context_department_no":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=context_department_no)),
        "context_index_code":
            tf.train.Feature(
                bytes_list=tf.train.BytesList(value=context_index_code)),
        "context_index_group_no":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=context_index_group_no)),
        "context_section_no":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=context_section_no)),
        "context_garment_group_no":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=context_garment_group_no)),
        "context_timestamp":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=context_timestamp)),
        "context_price":
            tf.train.Feature(
                float_list=tf.train.FloatList(value=context_price)),
        "context_sales_channel_id":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=context_sales_channel_id)),
        "context_fn":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=context_fn)),
        "context_active":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=context_active)),
        "context_club_member_status":
            tf.train.Feature(
                bytes_list=tf.train.BytesList(value=context_club_member_status)),
        "context_fashion_news_frequency":
            tf.train.Feature(
                bytes_list=tf.train.BytesList(value=context_fashion_news_frequency)),
        "context_age":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=context_age)),
        "label_article_id":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=label_article_id)),
        "label_product_code":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=label_product_code)),
        "label_product_type_no":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=label_product_type_no)),
        "label_graphical_appearance_no":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=label_graphical_appearance_no)),
        "label_colour_group_code":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=label_colour_group_code)),
        "label_perceived_colour_value_id":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=label_perceived_colour_value_id)),
        "label_perceived_colour_master_id":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=label_perceived_colour_master_id)),
        "label_department_no":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=label_department_no)),
        "label_index_code":
            tf.train.Feature(
                bytes_list=tf.train.BytesList(value=label_index_code)),
        "label_index_group_no":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=label_index_group_no)),
        "label_section_no":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=label_section_no)),
        "label_garment_group_no":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=label_garment_group_no))
    }
    
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    examples.append(tf_example)

  return examples


def generate_examples_from_timelines(timelines,
                                     article_df,
                                     min_timeline_len=3,
                                     max_context_len=100,
                                     train_data_fraction=0.9,
                                     random_seed=None,
                                     shuffle=True):
  """Convert user timelines to tf examples.
  Convert user timelines to tf examples by adding all possible context-label
  pairs in the examples pool.
  Args:
    timelines: The user timelines to process.
    article_df: The dataframe of all articles.
    min_timeline_len: The minimum length of timeline. If the timeline length is
      less than min_timeline_len, empty examples list will be returned.
    max_context_len: The maximum length of the context. If the context history
      length is less than max_context_length, features will be padded with
      default values.
    train_data_fraction: Fraction of training data.
    random_seed: Seed for randomization.
    shuffle: Whether to shuffle the examples before splitting train and test
      data.
  Returns:
    train_examples: TF example list for training.
    test_examples: TF example list for testing.
  """
  examples = []
  articles_dict = generate_articles_dict(article_df)
  progress_bar = tf.keras.utils.Progbar(len(timelines))
  for timeline in timelines.values():
    if len(timeline) < min_timeline_len:
      progress_bar.add(1)
      continue
    single_timeline_examples = generate_examples_from_single_timeline(
        timeline=timeline,
        articles_dict=articles_dict,
        max_context_len=max_context_len,)
    examples.extend(single_timeline_examples)
    progress_bar.add(1)
  # Split the examples into train, test sets.
  if shuffle:
    random.seed(random_seed)
    random.shuffle(examples)
  last_train_index = round(len(examples) * train_data_fraction)

  train_examples = examples[:last_train_index]
  test_examples = examples[last_train_index:]
  return train_examples, test_examples

def write_vocab_json(vocab, filename):
  """Write generated article vocabulary to specified file."""
  with open(filename, "w", encoding="utf-8") as jsonfile:
    json.dump(vocab, jsonfile, indent=2)


def write_vocab_txt(vocab, filename):
  with open(filename, "w", encoding="utf-8") as f:
    for item in vocab:
      f.write(str(item) + "\n")


def generate_datasets(extracted_data_dir,
                      output_dir,
                      min_timeline_length,
                      max_context_length,
                      build_vocabs=True,
                      train_data_fraction=0.9,
                      train_filename=OUTPUT_TRAINING_DATA_FILENAME,
                      test_filename=OUTPUT_TESTING_DATA_FILENAME
                      ):
  """Generates train and test datasets as TFRecord, and returns stats."""
  logging.info("Reading data to dataframes.")
  train_ds, article_ds = read_data(extracted_data_dir)
  logging.info("Generating article user timelines.")
  stats_dict = {}
  for col in CATEG_CUSTOMER_COLS:
      stats_dict[col.lower()] = sorted(train_ds[col].unique())
  for col in article_ds.columns:
    stats_dict[col.lower()] = sorted(article_ds[col].unique())
  timelines, articles_count = convert_to_timelines(train_ds)
  logging.info("Generating train and test examples.")
  train_examples, test_examples = generate_examples_from_timelines(
      timelines=timelines,
      article_df=article_ds,
      min_timeline_len=min_timeline_length,
      max_context_len=max_context_length,
      train_data_fraction=train_data_fraction)

  if not tf.io.gfile.exists(output_dir):
    tf.io.gfile.makedirs(output_dir)
  logging.info("Writing generated training examples.")
  train_file = os.path.join(output_dir, train_filename)
  train_size = write_tfrecords(tf_examples=train_examples, filename=train_file)
  logging.info("Writing generated testing examples.")
  test_file = os.path.join(output_dir, test_filename)
  test_size = write_tfrecords(tf_examples=test_examples, filename=test_file)
  stats = {
      "train_size": train_size,
      "test_size": test_size,
      "train_file": train_file,
      "test_file": test_file,
      "stats_dict": stats_dict
  }
  stats_path = os.path.join(output_dir, "stats_dict.pkl")
  stats_file = open(stats_path, "wb")
  pickle.dump(stats, stats_file)
  stats_file.close()
  return stats

def write_tfrecords(tf_examples, filename):
  """Writes tf examples to tfrecord file, and returns the count."""
  with tf.io.TFRecordWriter(filename) as file_writer:
    length = len(tf_examples)
    progress_bar = tf.keras.utils.Progbar(length)
    for example in tf_examples:
      file_writer.write(example.SerializeToString())
      progress_bar.add(1)
    return length

def main(_):
  logging.info("Downloading and extracting data.")
  stats = generate_datasets(
      extracted_data_dir=CSV_DIR,
      output_dir=FLAGS.output_dir,
      min_timeline_length=FLAGS.min_timeline_length,
      max_context_length=FLAGS.max_context_length,
      build_vocabs=FLAGS.build_vocabs,
      train_data_fraction=FLAGS.train_data_fraction,
  )
  logging.info("Generated dataset: %s", stats)

if __name__ == "__main__":
  define_flags()
  app.run(main)
