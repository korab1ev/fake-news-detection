# fake-news-detection
Binary classification task for Kontur summer 2022 internship
Файлы: 
- ``config.yaml``
- ``data.py``
- ``model.py``
- ``train.py``

относятся к решению на основе нижеупомянутых 6-ти моделей. (Для них пайплайн одинаковый с точностью до замены названия дообучаемой модели)


### Результаты


| model  | best train epoch | best model F1 | best model accuracy |
|---|---|---|---|
| DeepPavlov/rubert-base-cased | 2 | 0.94447 | 0.94444 |
| DeepPavlov/rubert-base-cased-sentence | 5 | 0.94446 | 0.94444 |
| DeepPavlov/bert-base-multilingual-cased-sentence | 1 | 0.92357 | 0.92361 |
| bert-base-multilingual-cased | 13 | 0.92013 | 0.92013 |
| bert-base-multilingual-uncased | 7 | 0.92018 | 0.92013 |
| cointegrated/rubert-tiny | 5 | 0.90111 | 0.90104 |

Папка images содержит результаты сходимости каждой fine-tuned модели.


### Результаты простых методов 

| Vectorizing  | Classifier | Accuracy | F1 |
|---|---|---|---|
| BoW | LogisticRegression | 0.84895 | 0.84895 |
| BoW | RandomForest | 0.78819 | 0.78819 |
| Tf-idf | LogisticRegression | 0.85763 | 0.85763 |
| Tf-idf | RandomForest | 0.75347 | 0.75347 |

Рассмотрел их тоже так как было интересно ответить на вопрос, а есть ли вообще смысл обучать тяжеловесные модели, когда существуют простые подходы.  
Разница в перфомансе этих методов и моделей на основе BERT все-таки большая.

Word2vec обучать не стал, мало данных (поэтому с большой вероятностью он будет хуже обычного bag of words).
