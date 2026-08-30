# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This project builds a classification model that predicts whether a person's income is above or below $50K per year using the Adult Census Income data. The model is a scikit-learn RandomForestClassifier with 100 trees, a max depth of 15, and random_state set to 42. Categorical features are one-hot encoded, and the salary label is converted to a binary target. The trained model and encoder are saved with pickle and used by a FastAPI app for predictions.

## Intended Use

This model is meant for a class project that shows an end-to-end machine learning pipeline. That includes training, checking performance on data slices, unit testing, continuous integration, and a simple REST API. The model predicts income category (`>50K` or `<=50K`) from census features. It should not be used for real hiring, credit, or other high-stakes decisions.

## Training Data

The training data comes from the cleaned `data/census.csv` file in this repository. Features include age, workclass, education, occupation, relationship, race, sex, capital gain and loss, hours per week, native country, and related fields. The label column is `salary` with values `<=50K` and `>50K`. The data is split into 80% training and 20% testing with `train_test_split` and `random_state=42`. The encoder, label binarizer, and model are fit only on the training set.

## Evaluation Data

The evaluation data is the 20% held-out test set from the same split. The test set is transformed with the encoder and label binarizer that were fit on the training data. Performance is also checked on slices of the test set for each unique value of the categorical features. Those slice results are saved in `slice_output.txt`.

## Metrics

The model is evaluated with precision, recall, and F1 score for the positive class `>50K`.

On the held-out test set from the final training run, the results were:

- **Precision:** 0.7918
- **Recall:** 0.5786
- **F1:** 0.6686

Slice-level precision, recall, and F1 for each categorical feature value are listed in `slice_output.txt`.

## Ethical Considerations

Income prediction models can show bias that already exists in the data. Performance may differ across groups such as race, sex, or native country. Slice metrics are included so those differences can be reviewed. This model should not be the only input for decisions that affect people's jobs, credit, or access to services.

## Caveats and Recommendations

The random forest settings were chosen for a stable class project baseline, not full hyperparameter tuning. The dataset is imbalanced, which helps explain why recall is lower than precision. Rare or unseen category values at prediction time may be handled less well because unknown categories are ignored by the encoder. Future improvements could include tuning the decision threshold, reviewing slice metrics more closely for fairness, and rechecking performance when the data is updated.
