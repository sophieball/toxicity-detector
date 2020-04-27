# Current Infrastructure Setup (internal)

We use the GHTorrent MongoDB for the sources of issues and issue_comments.

The goal is to classify all issues and comments from the newest to the oldest.

## Database layout

Since I don't want to modify the original GHTorrent collections, there are two new collections that serve as todo list and store results: `christian_toxic_issues` and `christian_toxic_issue_comments`. Both collections have documents for each document in the GhTorrent collections with corresponding `_id` and also share the `updated_at` field.

The following script can import the issues into the new collections

```sh
mongo ghtorrent --eval "db.issues.find({updated_at: {\$regex: '^2020-'}},{updated_at:1}).sort({updated_at:-1}).forEach(function(i){db.christian_toxic_issues.insert(i)})"
mongo ghtorrent --eval "db.issue_comments.find({updated_at: {\$regex: '^2020-'}},{updated_at:1,repo:1,owner:1,issue_id:1}).sort({updated_at:-1}).forEach(function(i){db.christian_toxic_issue_comments.insert(i)})"
```

## Running the classifier

With `src/classify_comments.py`, we go over all issues in the `christian_toxic_issues` and `christian_toxic_issue_comments` collections from newest to oldest by the `updated_at` field.

In `config.py` the database credentials and the perspective API key need to be provided.

Multiple copies of the script can be run in parallel.

It adds results to the `toxicity.<version>` field of each collection.
In addition, `toxicity.manual.toxic` may be used to store human labels (e.g., set through a user interface), and other fields in `toxicity.manual` may provide additional information about the label.

Result layouts may differ between different revisions of the classifier, but generally `toxicity.<version>.score` should be 1 for toxic comments and 0 otherwise. Other fields may store inputs to the prediction, such as details from the perspective calls or whether the text was considered to be English.

Different versions of the classifier may perform different tasks and may process text differently.
* Version `v1`: Perspective toxicity score and politeness score with pretrained classifier. No preprocessing of the text.
* Version `v2`: Preprocessing of the text to remove markdown, markdown-code, and html. Tries to detect whether text is English in field `en`. Stores raw result from perspective API.

## Browsing results

`browser/main.py` is a simple flask application (run with `export FLASK_APP=main.py; flask run`) that shows the classification results:
* `/` some statistics about the classification progress and results (takes a while to compute)
* `/list/toxic` and `/list/toxic_comments` show a list of all issues/comments classified as toxic. From there links will get you to the specific issues, from where further links are provided to add ground-truth (manual) labels

