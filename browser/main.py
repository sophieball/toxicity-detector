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

VERSION="v2" 

app = Flask(__name__)
app.config['DEBUG'] = True

query = { "$and":[ 
		{"toxicity."+VERSION+".score":1}, 
		{"toxicity."+VERSION+".orig.persp_raw.detectedLanguages":["en"]},
		{"toxicity.v2.en":{"$gt": .001}}
	]}   


@app.route('/')
def dashboard():
	issues_analyzed = db.christian_toxic_issues.count({"toxicity."+VERSION:{"$exists":1}})
	issues_newest_not_analyzed = db.christian_toxic_issues.find_one({"toxicity."+VERSION:{"$exists":0}},sort=[("updated_at",-1)])["updated_at"]
	issues_newest_analyzed = db.christian_toxic_issues.find_one({"toxicity."+VERSION:{"$exists":1}},sort=[("updated_at",-1)])["updated_at"]
	issues_oldest_analyzed = db.christian_toxic_issues.find_one({"toxicity."+VERSION:{"$exists":1}},sort=[("updated_at",1)])["updated_at"]
	issues_toxic = db.christian_toxic_issues.count(query)
	comments_analyzed = db.christian_toxic_issue_comments.count({"toxicity."+VERSION:{"$exists":1}})
	comments_newest_not_analyzed = db.christian_toxic_issue_comments.find_one({"toxicity."+VERSION:{"$exists":0}},sort=[("updated_at",-1)])["updated_at"]
	comments_newest_analyzed = db.christian_toxic_issue_comments.find_one({"toxicity."+VERSION:{"$exists":1}},sort=[("updated_at",-1)])["updated_at"]
	comments_oldest_analyzed = db.christian_toxic_issue_comments.find_one({"toxicity."+VERSION:{"$exists":1}},sort=[("updated_at",1)])["updated_at"]
	comments_toxic = db.christian_toxic_issue_comments.count(query)
	return render_template('./dashboard.html', 
		version = VERSION,
	    issues_analyzed=issues_analyzed,
	    issues_newest_not_analyzed=issues_newest_not_analyzed,
	    issues_newest_analyzed=issues_newest_analyzed,
	    issues_oldest_analyzed=issues_oldest_analyzed,
	    issues_toxic=issues_toxic,
	    comments_analyzed=comments_analyzed,
	    comments_newest_not_analyzed=comments_newest_not_analyzed,
	    comments_newest_analyzed=comments_newest_analyzed,
	    comments_oldest_analyzed=comments_oldest_analyzed,
	    comments_toxic=comments_toxic,
		)

 

@app.route('/list/toxic')
def list_toxic():
	tissues = db.christian_toxic_issues.find(query).sort([("updated_at",-1)])
	return render_template('./list.html', tissues=tissues, link="issue", deep_get=deep_get, what="issues")

@app.route('/list/toxic_comments')
def list_toxiccomments():
	tissues = db.christian_toxic_issue_comments.find(query).sort([("updated_at",-1)])
	return render_template('./list.html', tissues=tissues, link="comment", deep_get=deep_get, what="comments")

@app.route('/issue/<issueid>')
def show_issue(issueid, commentid=0):
	issue = db.issues.find_one({"_id":ObjectId(issueid)})
	tissue = db.christian_toxic_issues.find_one({"_id":ObjectId(issueid)})
	tissue["labels"] = label_buttons("/label/issue/", tissue)

	comments_cursor = db.issue_comments.find({"owner":issue["owner"],
		                "repo": issue["repo"],
		                "issue_id": issue["number"]})
	comments = []
	for c in comments_cursor:
		t = db.christian_toxic_issue_comments.find_one({"_id":c["_id"]})
		if t:
			if "toxicity" in t:
				c["toxicity"] = t["toxicity"]
		c["labels"] = label_buttons("/label/comment/", c)
		comments += [c]

	return render_template('./issue.html', issue=issue, tissue=tissue, is_toxic=is_toxic, comments = comments, render_label_buttons=render_label_buttons)

@app.route('/comment/<commentid>')
def show_comment(commentid):
	comment = db.issue_comments.find_one({"_id":ObjectId(commentid)})
	issue = db.issues.find_one({"owner":comment["owner"],
		                "repo": comment["repo"],
		                "number": comment["issue_id"]})
	if not issue:
		return "<p>Issue not found in database <a href=\""+comment["html_url"]+"\">"+comment["html_url"]+"</a></p>"
	return redirect("/issue/"+str(issue["_id"])+"#"+str(comment["id"]))


@app.route('/label/issue/<issueid>/<label>')
def label_issue(issueid, label):
	return label_entry("christian_toxic_issues",issueid,label)

@app.route('/label/comment/<commentid>/<label>')
def label_comment(commentid, label):
	return label_entry("christian_toxic_issue_comments",commentid,label)

def label_entry(table, eid, label):
	score = 0
	if label=="toxic":
		score = 1
	reason = label
	r=db[table].find_one_and_update({"_id":ObjectId(eid)},{"$set":{"toxicity.manual.score":score,"toxicity.manual.reason":reason}})
	if not r:
		return str(eid)+" not found in "+table
	return str(eid)+" updated"


labels = [["Confirm toxic","toxic"], 
          ["Not toxic -- Non english", "not-english"], 
          ["Not toxic -- Selfdirected", "self-directed"],
          ["Not toxic -- Owner", "owner"],
          ["Not toxic -- Mild/colloquial", "mild"],
          ["Not toxic -- Other", "other"]]
# gets data for buttons
# [url, label_text, is_selected]
def label_buttons(link, titem):
	current_label = deep_get(titem,"toxicity.manual.reason")
	result = []
	for l in labels:
		result += [[
			link + str(titem["_id"])+"/"+l[1],
			l[0],
			l[1]==current_label
		]]
	return result

def render_label_buttons(label_buttons):
	result = ""
	for b in label_buttons:
		result += "<a href=\""+b[0]+"\" class=\"labelbtn labelbtn_"+str(b[2])+"\">"+b[1]+"</a> "
	return result


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
	return te["toxicity"][VERSION]["score"]==1

from functools import reduce
def deep_get(dictionary, keys, default=None):
	return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("."), dictionary)
