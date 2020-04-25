from flask import Flask,redirect
from flask import render_template
import config
import pymongo
from bson.objectid import ObjectId

def connect_to_database():
	mongo_name = config.mongo["user"]
	mongo_password = config.mongo["passwd"]

	# Connect to the Mongo Database
	client = pymongo.MongoClient()
	db = client[config.mongo["db"]]
	db.authenticate(name=mongo_name, password=mongo_password)
	return db
db = connect_to_database()
 

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/list/toxic')
def list_toxic():
	tissues = db.christian_toxic_issues.find({"toxicity.v1.score":1}).sort([("updated_at",-1)])
	return render_template('./list.html', tissues=tissues, link="issue")

@app.route('/list/toxic_comments')
def list_toxiccomments():
	tissues = db.christian_toxic_issue_comments.find({"toxicity.v1.score":1}).sort([("updated_at",-1)])
	return render_template('./list.html', tissues=tissues, link="comment")

@app.route('/issue/<issueid>')
def show_issue(issueid, commentid=0):
	issue = db.issues.find_one({"_id":ObjectId(issueid)})
	tissue = db.christian_toxic_issues.find_one({"_id":ObjectId(issueid)})

	comments_cursor = db.issue_comments.find({"owner":issue["owner"],
		                "repo": issue["repo"],
		                "issue_id": issue["number"]})
	comments = []
	for c in comments_cursor:
		t = db.christian_toxic_issue_comments.find_one({"_id":c["_id"]})
		if t:
			if "toxicity" in t:
				c["toxicity"] = t["toxicity"]
		comments += [c]

	return render_template('./issue.html', issue=issue, tissue=tissue, is_toxic=is_toxic, comments = comments)

@app.route('/comment/<commentid>')
def show_comment(commentid):
	comment = db.issue_comments.find_one({"_id":ObjectId(commentid)})
	issue = db.issues.find_one({"owner":comment["owner"],
		                "repo": comment["repo"],
		                "number": comment["issue_id"]})
	if not issue:
		return "<p>Issue not found in database <a href=\""+comment["html_url"]+"\">"+comment["html_url"]+"</a></p>"
	return redirect("/issue/"+str(issue["_id"])+"#"+str(comment["id"]))

def get_owner_repo_issueid(tcomment):
	if "repo" in tcomment:
		return [tcomment["owner"],tcomment["repo"],tcomment["issue_id"]]
	else:
		comment = db.issue_comments.find_one({"_id":ObjectId(tcomment["_id"])})
		return [comment["owner"],comment["repo"],comment["issue_id"]]

def is_toxic(te):
	if not te: 
		return False
	if not "toxicity" in te:
		return False
	return te["toxicity"]["v1"]["score"]==1