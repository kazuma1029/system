import random
import pandas as pd
import torch
from transformers import BertJapaneseTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from fugashi import Tagger
from collections import Counter
from math import log
import shutil
import os

import torch

import transformers

# スクリプト位置から1つ上のディレクトリを基準にする（workspace のルート側）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

print("Transformers version:", transformers.__version__)
print("Transformers location:", transformers.__file__)


# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# elif torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
device = torch.device("cpu")

print(f"Using device: {device}")


hikensya_num = input("被験者番号を入力してください：")
print(f"被験者{hikensya_num}の嗜好を基にBERTをファインチューニングします．")

# choice = input("分類モードを選択してください（1: アンダーサンプリングなし, 2: アンダーサンプリングあり： ")
# ステップ1: 好みの映画と好みでない映画のIDを読み込む
liked_movies_file = f"./movies/{hikensya_num}_liked_movies.txt"
disliked_movies_file = f"./movies/{hikensya_num}_disliked_movies.txt"

with open(liked_movies_file, 'r') as file:
    liked_movie_ids = [line.strip() for line in file.readlines()]

with open(disliked_movies_file, 'r') as file:
    disliked_movie_ids = [line.strip() for line in file.readlines()]

#レビュー文のランキング出力用
def output_reviews_sorted_by_score(reviews, total_scores, output_file):
    tagger = Tagger()
    review_scores = []

    # 各レビューのスコアを計算
    for review in reviews:
        score = 0
        nouns = [word.surface for word in tagger(review) if '名詞' in word.feature]
        for noun in nouns:
            score += total_scores.get(noun, 0)
        review_scores.append((review, score))

    # スコア順にソート
    review_scores.sort(key=lambda x: x[1], reverse=True)

    # テキストファイルに出力
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("レビュー文, スコア\n")
        for review, score in review_scores:
            f.write(f"{review}, {score:.6f}\n")

    print(f"レビュー文をスコア順に {output_file} に保存しました。")

# 使用例: 全レビュー文を出力
all_reviews_file = f"./ランキング/{hikensya_num}_all_reviews_sorted.txt"
# ステップ2: レビューを読み込む関数を定義
def load_reviews(movie_ids, label):
    reviews = []
    labels = []
    for movie_id in movie_ids:
        file_path = os.path.join(BASE_DIR, '極性付きレビューファイル', f"{movie_id}.xlsx")
        try:
            data = pd.read_excel(file_path)
            # 一列目がレビュー文、二列目が極性の列
            for _, row in data.iterrows():
                if row.iloc[1] == 1:  # 二列目の値が"1"の場合のみ処理
                    reviews.append(row.iloc[0])  # 一列目のレビュー文を追加
                    labels.append(label)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    return reviews, labels

# TF-IDF計算関数
def calculate_tfidf(all_reviews, total_movies):
    term_movie_count = Counter()
    tf_values = []

    for reviews in all_reviews:
        doc_terms = Counter()
        for review in reviews:
            words = review.split()
            term_count = Counter(words)
            doc_terms.update(term_count)
        tf_values.append(doc_terms)

        unique_terms = set(doc_terms.keys())
        for term in unique_terms:
            term_movie_count[term] += 1

    idf_values = {term: log(total_movies / (1 + count)) for term, count in term_movie_count.items()}

    tfidf_scores = []
    for tf in tf_values:
        tfidf_scores.append({term: tf_val * idf_values[term] for term, tf_val in tf.items()})

    return tfidf_scores, idf_values

# 名詞抽出とTF-IDF計算
def extract_reviews_and_nouns(movie_ids, total_movies, output_file):
    tagger = Tagger()  # M1/M2 MacなどHomebrew経由で入れた場合の例

    all_reviews = []
    all_nouns = []

    for movie_id in movie_ids:
        #あらかじめ映画の全レビュー文を極性を判別したexcelファイルを読み込み
        file_path = f"./極性付きレビューファイル/{movie_id}.xlsx"
        try:
            data = pd.read_excel(file_path)
            positive_reviews = data[data.iloc[:, 1] == 1].iloc[:, 0].dropna().tolist()
            movie_reviews = []
            for review in positive_reviews:
                nouns = ' '.join([word.surface for word in tagger(review) if '名詞' in word.feature])
                if nouns:
                    all_reviews.append(review)
                    movie_reviews.append(nouns)
            #映画mの各レビューに名詞が含まれていればその各名詞を収集
            if movie_reviews:
                all_nouns.append(movie_reviews)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    if all_nouns:
        tfidf_scores, idf_values = calculate_tfidf(all_nouns, total_movies)

        total_scores = Counter()
        for doc_scores in tfidf_scores:
            for term, score in doc_scores.items():
                total_scores[term] += score

        sorted_nouns = total_scores.most_common()

       # 全単語ランキングをファイルに出力
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("単語, スコア\n")
            for term, score in sorted_nouns:
                f.write(f"{term}, {score:.6f}\n")
        print(f"名詞ランキングを {output_file} に保存しました。")

    return all_reviews, sorted_nouns,total_scores

#レビュースコア計算
def calculate_review_scores(reviews, total_scores):
    tagger = Tagger()
    review_scores = []

    for review in reviews:
        score = 0
        nouns = [word.surface for word in tagger(review) if '名詞' in word.feature]
        for noun in nouns:
            score += total_scores.get(noun, 0)
        review_scores.append((review, score))

    return review_scores

#outputディレクトリ消去用
def clean_output_directory(output_dir):
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
            print(f"ディレクトリ {output_dir} を削除しました。")
        except Exception as e:
            print(f"ディレクトリ {output_dir} の削除中にエラーが発生しました: {e}")
    else:
        print(f"ディレクトリ {output_dir} は存在しません。")

#乱数設定用
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 好みの映画と好みでない映画のレビューを読み込む
liked_reviews, liked_nouns, liked_total_scores = extract_reviews_and_nouns(liked_movie_ids, total_movies=10, output_file=f"./ランキング/{hikensya_num}_liked_名詞ランキング.txt")
disliked_reviews, disliked_nouns, disliked_total_scores = extract_reviews_and_nouns(disliked_movie_ids, total_movies=10, output_file=f"./ランキング/{hikensya_num}_disliked_名詞ランキング.txt")

liked_reviews_file = f"./ランキング/{hikensya_num}_liked_reviews.txt"
disliked_reviews_file = f"./ランキング/{hikensya_num}_disliked_reviews.txt"
output_reviews_sorted_by_score(liked_reviews,liked_total_scores,liked_reviews_file)
output_reviews_sorted_by_score(disliked_reviews,disliked_total_scores,disliked_reviews_file)
liked_labels=[]
disliked_labels=[]

for r in liked_reviews:
    liked_labels.append(1)

for r in disliked_reviews:
    disliked_labels.append(0)


# #レビュー数が異なる場合アンダーサンプリング
# for ran in range(1,11):
#     set_seed(42 + ran)
#     if len(liked_reviews) > len(disliked_reviews):
#         excess = len(liked_reviews) - len(disliked_reviews)
#         indices_to_remove = torch.randperm(len(liked_reviews))[:excess]
#         liked_reviews = [review for i, review in enumerate(liked_reviews) if i not in indices_to_remove]
#         liked_labels = liked_labels[:len(liked_reviews)]
#     elif len(disliked_reviews) > len(liked_reviews):
#         excess = len(disliked_reviews) - len(liked_reviews)
#         indices_to_remove = torch.randperm(len(disliked_reviews))[:excess]
#         disliked_reviews = [review for i, review in enumerate(disliked_reviews) if i not in indices_to_remove]
#         disliked_labels = disliked_labels[:len(disliked_reviews)]


#     # ステップ3: データを統合
#     all_reviews=liked_reviews+disliked_reviews
#     all_labels=liked_labels+disliked_labels
#     # ステップ4: データを分割（全レビューをファインチューニングに使用し、最初の10件をテストに使用）
#     train_texts = all_reviews
#     train_labels = all_labels

# 元データのバックアップを作成（ループ前）
original_liked_reviews = liked_reviews[:]
original_disliked_reviews = disliked_reviews[:]
original_liked_labels = liked_labels[:]
original_disliked_labels = disliked_labels[:]

# モデル作成ループ
for ran in range(2, 4):
    set_seed(42 + ran)

    # コピー（毎回初期状態に戻す）
    current_liked_reviews = original_liked_reviews[:]
    current_disliked_reviews = original_disliked_reviews[:]
    current_liked_labels = original_liked_labels[:]
    current_disliked_labels = original_disliked_labels[:]

    # 正しくアンダーサンプリング
    if len(current_liked_reviews) > len(current_disliked_reviews):
        excess = len(current_liked_reviews) - len(current_disliked_reviews)
        indices_to_remove = set(torch.randperm(len(current_liked_reviews))[:excess].tolist())
        current_liked_reviews = [r for i, r in enumerate(current_liked_reviews) if i not in indices_to_remove]
        current_liked_labels = [l for i, l in enumerate(current_liked_labels) if i not in indices_to_remove]

    elif len(current_disliked_reviews) > len(current_liked_reviews):
        excess = len(current_disliked_reviews) - len(current_liked_reviews)
        indices_to_remove = set(torch.randperm(len(current_disliked_reviews))[:excess].tolist())
        current_disliked_reviews = [r for i, r in enumerate(current_disliked_reviews) if i not in indices_to_remove]
        current_disliked_labels = [l for i, l in enumerate(current_disliked_labels) if i not in indices_to_remove]

    # データ統合
    all_reviews = current_liked_reviews + current_disliked_reviews
    all_labels = current_liked_labels + current_disliked_labels

    # 学習・テストデータ作成
    all_reviews = current_liked_reviews + current_disliked_reviews
    all_labels = current_liked_labels + current_disliked_labels
    train_texts = all_reviews
    train_labels = all_labels
    test_texts = all_reviews[:10]
    test_labels = all_labels[:10]

    # train_texts, val_texts, train_labels, val_labels = train_test_split(
    #     all_reviews, all_labels, test_size=0.2, random_state=42
    # )

    # ステップ5: トークナイズ
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')

    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=512)

    print(f"被験者{hikensya_num}の好みの映画のレビュー数は{len(current_liked_reviews)}です．")
    print(f"被験者{hikensya_num}の好みでない映画のレビュー数は{len(current_disliked_reviews)}です．")
    print(f"全レビュー数は{len(all_reviews)}です．")

    # ステップ6: PyTorch Datasetの作成
    class MovieReviewDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

    train_dataset = MovieReviewDataset(train_encodings, list(train_labels))
    test_dataset = MovieReviewDataset(test_encodings, list(test_labels))


    #分類器の層の初期値のランダム初期化の値を固定
    set_seed(42)

    # ステップ5: 事前学習済みBERTモデルをロード
    model = BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese', num_labels=2)
    # ステップ6: トークナイズ
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')

    # 分類器の初期値を確認（オプション）
    print("初期分類器の重み:", model.classifier.weight)
    print("初期分類器のバイアス:", model.classifier.bias)

    # ステップ8: 評価指標を定義
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # ステップ9: トレーニング引数
    training_args = TrainingArguments(
        output_dir=f'./output/{hikensya_num}_all',    # 出力ディレクトリ
        num_train_epochs=3,              # トレーニングエポック数
        per_device_train_batch_size=16,  # トレーニング時のバッチサイズ
        per_device_eval_batch_size=64,   # 評価時のバッチサイズ
        warmup_steps=500,                # 学習率スケジューラのウォームアップステップ数
        weight_decay=0.01,               # 重み減衰（L2正則化）
        # logging_dir='./logs',            # ログ保存ディレクトリ
        logging_dir='C:/Users/kazuma/logs',
        logging_steps=10,                # ログ出力間隔（ステップ数）
        evaluation_strategy="epoch",     # 各エポック終了時に評価
        save_strategy="epoch",           # 各エポック終了時にモデルを保存
        load_best_model_at_end=True      # 最良モデルを最後にロード
    )

    # ステップ10: Trainerの設定
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # ステップ11: モデルをトレーニング
    trainer.train()

    # ステップ12: モデルを保存

    trainer.save_model(f"./models/{hikensya_num}_allmodel_rus_{ran}")
    tokenizer.save_pretrained(f"./models/{hikensya_num}_allmodel_rus_{ran}")        

    # モデル作成後に出力ディレクトリを削除
    clean_output_directory(f"./output/{hikensya_num}_all")

    # ステップ13: モデルを評価（テストデータで）
    predictions = trainer.predict(test_dataset)
    eval_results = compute_metrics(predictions)
    print("テストデータでの評価結果:", eval_results)
