import pandas as pd
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import tweepy
from application_tokens import *
import json
from collections import Counter
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib import rcParams
import seaborn as sns
from textwrap import wrap
from wordcloud import WordCloud, STOPWORDS
from PIL import Image, ImageDraw, ImageFont

print('Load sentiment intensity analyzer and spacy model...')

analyzer = SentimentIntensityAnalyzer()
nlp = spacy.load('en')

#define function to run sentiment analysis and named entity recognition on given text
def process_text(text):
	vs = analyzer.polarity_scores(text)
	doc = nlp(text)
	return vs, doc.ents

print('Set up tweepy...')

#set up tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
try:
	redirect_url = auth.get_authorization_url()
except tweepy.TweepError:
	print('Error! Failed to get request token.')
api = tweepy.API(auth)

def get_tweets(screen_name, count):
    new_tweets = api.user_timeline(screen_name=screen_name, count=count, tweet_mode='extended')
    new_tweets = [{"created_at":i.created_at.strftime("%Y-%m-%d %H:%M:%S"),"id": i.id,'retweet_count':i.retweet_count,'favorite_count':i.favorite_count,'text':i.full_text} for i in new_tweets]
    return new_tweets

ents = {}
counts = {}
screen_name = 'realDonaldTrump'
num_tweets = 200

for i in get_tweets(screen_name, num_tweets):
    vs, tweet_ents = process_text(i['text'])
    #accumulate named entities
    for ent in tweet_ents:
        if ent.text in counts:
            counts[ent.text] += 1
        else:
            ents[ent.text] = ent.label_
            counts[ent.text] = 1

print('Get last 200 tweets and save to json...')

#check Trump's last 200 tweets and save to json
with open('new_tweets.json','wb') as f:
	f.write(json.dumps(get_tweets(screen_name, num_tweets)).encode('utf-8'))

print('Load tweets from past day and retrieve tweet text, entities, and sentiment...')

#load tweets from past day and retrieve tweet text, entities, and sentiment
ents_dict = {}
counts_dict = {}
retweet_count = 0
favorite_count = 0
full_text = ''
pos_text = ''
neg_text = ''
neu_text = ''
sentiment = []
entities = []
tweet_count = 0
avg_sentiment = 0
# collection of words and their cumulative sentiment
counter = Counter()
pos_count = Counter()
neg_count = Counter()
neu_count = Counter()
pos_sent = Counter()
neg_sent = Counter()
neu_sent = Counter()
#test wordcloud
wordcloud = WordCloud(width=1,height=1).generate('test')
#process text of tweets created today
for tweet in json.load(open('new_tweets.json')):
	if (tweet['created_at'][:10] == str(datetime.now().date())):
		tweet_count += 1
		retweet_count += tweet['retweet_count']
		favorite_count += tweet['favorite_count']
		tweet_text = tweet['text']
		re.sub('[!@#$]|http\S*', '', tweet_text)
		full_text += tweet_text

		#sentiment and list of entities
		vs, daily_ents = process_text(tweet_text)

		#Counter of words and their occurrence counts
		words = Counter(WordCloud.process_text(wordcloud, tweet_text))
		#positive
		if vs['compound'] > 0:
			pos_text += ' ' + tweet_text
			pos_count.update(words)
			for key in words.keys():
				pos_sent.update(Counter({key:vs['compound']*words[key]}))
		else:
			#negative
			if vs['compound'] < 0:
				neg_text += tweet_text
				neg_count.update(words)
				for key in words.keys():
					neg_sent.update(Counter({key:vs['compound']*words[key]}))
			#neutral
			else:
				neu_text += tweet_text
				neu_count.update(words)
				for key in words.keys():
					neu_sent.update(Counter({key:vs['compound']*words[key]}))

		tweet_dict = {}
		for word in words.keys():
			if word in tweet_dict.keys():
				tweet_dict[word] += vs['compound'] * words[word]
			else:
				tweet_dict[word] = vs['compound'] * words[word]
		#update counter with sentiment of this tweet, multiplied by the number of occurrences
		counter.update(Counter(tweet_dict))
		
		#accumulate named entities
		for ent in daily_ents:
			if ent.text in counts_dict:
				counts_dict[ent.text] += 1
			else:
				ents_dict[ent.text] = ent.label_
				counts_dict[ent.text] = 1
		sentiment.append(vs['compound'])
		avg_sentiment += vs['compound']
		entities.append('\n'.join(str(e) for e in daily_ents))
avg_sentiment /= tweet_count

print('Generate text for tweet text and visualization text...')

#generate text for tweet text and visualization text
avg_sent=np.mean(sentiment)
dev_sent=np.std(sentiment)
def sentiment_sentence(mean,std):
    m_word = ["very negative","negative","neutral","positive","very positive"]
    m_word2 = ["overwhelmingly negative", "generally negative", "", "generally positive","overwhelmingly positive"]
    d_word = ["consistently","mixed"]
    score=[2,0]
    
    if mean>.6:
        score[0]=4
    elif mean<-.6:
        score[0]=0
    elif mean>.2:
        score[0]=3
    elif mean<-.2:
        score[0]=1
        
    if std>.5:
        score[1]=1
        
    base="The sentiment of his tweets was "
    
    if score[1]==0:
        base+=(d_word[0]+" "+m_word[score[0]])
    else:
        if score[0]==2:
            base+=d_word[1]
        else:
            base+=(d_word[1]+", but "+m_word2[score[0]])
    return base+"."





print('Get Trump approval data from FiveThirtyEight...')

#get Trump approval data from FiveThirtyEight
trump_approval_data = pd.read_csv('https://projects.fivethirtyeight.com/trump-approval-data/approval_topline.csv')
#change 'modeldate' datatype to datetime.date
for index,row in trump_approval_data.iterrows():
    trump_approval_data.at[index,'modeldate'] = datetime.strptime(row['modeldate'], '%m/%d/%Y').date()
#get approval estimates from the subgroup 'All polls'
approval_estimates = trump_approval_data[trump_approval_data['subgroup'] == 'All polls'][['modeldate','approve_estimate','disapprove_estimate']]
#get today's approval rating
current_approval = approval_estimates[approval_estimates['modeldate'] == datetime.now().date()]['approve_estimate'].values[0]

thirty_day = approval_estimates[(approval_estimates['modeldate'] >= datetime.now().date()-timedelta(days=30)) & (approval_estimates['modeldate'] < datetime.now().date())][['approve_estimate']]
approval_sentences = ["His approval rating (" + str(round(current_approval,1)) + "%) "+i+"." for i in ["is at a 30-day low","is lower than what it has been recently", "is similar to what it has been recently", "is higher than what it has been recently", "is at a 30-day high"]]
def approval_sentence(thirty_day,cur,sentences):
    thirds = sorted(thirty_day.approve_estimate.quantile([0,1/3.0,2/3.0,1]).tolist())
    for i in range(len(thirds)):
        if cur<thirds[i]:
            return sentences[i]
    return sentences[-1]
approval_str = approval_sentence(thirty_day, current_approval, approval_sentences)





#entities_list
text = list(ents_dict.keys())
labels = list(ents_dict.values())
data = {'text':text,'label':labels}
df2 = pd.DataFrame(data)
for index,row in df2.iterrows():
    df2.at[index,'count'] = counts[row['text']]
#sort by count
df2 = df2.sort_values(by=['count'], ascending=False)
arr = df2.head()['text'].values
entities_list = ''.join(str(e + ', ') for e in arr[:-1])
entities_list += 'and ' + arr[len(arr)-1] + '.'

count_str = 'Today, @' + screen_name + ' tweeted ' + str(tweet_count) + ' times.'
sentiment_str = sentiment_sentence(avg_sent,dev_sent)
entities_str = 'He mentioned ' + entities_list

final_str = count_str + ' ' + sentiment_str + ' ' + entities_str + ' ' + approval_str

print('Create barplot of retweets and favorites...')

#######barplot of retweets and favorites
#fig = plt.figure(figsize=(14.2, 7.12))
plt.figure(figsize=(8, 4),dpi=128)
ax = sns.barplot(y=['retweets','favorites'], x=[retweet_count,favorite_count],palette=['skyblue','lightcoral'],orient='h')

tick_prop = fm.FontProperties(fname='Libre_Franklin/LibreFranklin-Light.ttf',size=8)
ax_prop = fm.FontProperties(fname='Libre_Franklin/LibreFranklin-Light.ttf',size=8)
title_prop = fm.FontProperties(fname='Libre_Franklin/LibreFranklin-SemiBold.ttf',size=18)

ax.set_xlabel('', fontproperties=ax_prop, color='#555555')
plt.figtext(.5,.92, 'Tweet statistics of @' + screen_name + ' - ' + str(datetime.now().date()), fontproperties=title_prop, ha='center')

bar_prop = fm.FontProperties(fname='Libre_Franklin/LibreFranklin-SemiBold.ttf',size=16)

plt.figtext(.15,.64, str(retweet_count) + '\nretweets', fontproperties=bar_prop,color='white')
plt.figtext(.15,.26, str(favorite_count) + '\nfavorites', fontproperties=bar_prop,color='white')

plt.xticks(fontproperties=tick_prop,color='#555555')
plt.yticks(size=0)
plt.savefig('stats_plt.png')
stats_img = Image.open('stats_plt.png')

#stats_transparent = Image.new('RGBA', (1024, 512), (255,255,255,254))

#stats_transparent = Image.new('RGBA', (1024, 512), (255,255,255,254))
#stats_img = Image.open('stats_plt.png')
#new_box = (2,0, stats_img.size[0]+2,stats_img.size[1])
#stats_transparent.paste(stats_img,new_box)
#stats_transparent.save('stats_plt.png')

print('Create violin plot of sentiment...')

#####violin plot of sentiment
prop = fm.FontProperties(fname='Libre_Franklin/LibreFranklin-SemiBold.ttf',size=28)

rcParams['text.color'] = '#555555'
rcParams['axes.titlesize'] = 'large'
rcParams['axes.titleweight'] = 'bold'

#fig, ax = plt.subplots(figsize=(10.25,7))
fig, ax = plt.subplots(figsize=(8,4),dpi=128)

# sep is the point where the separation should occur
sep = 0
plt.axvline(x=sep, color='#4c4c4c')

sns.violinplot(ax=ax, x=sentiment)

#obtain path of violin surrounding
path = ax.collections[0].get_paths()[0]
path_neg = path.clip_to_bbox([-2,-2,0,1], inside=True)
patch_neg = matplotlib.patches.PathPatch(path_neg, facecolor='lightcoral', lw=2, ec='#4c4c4c')
path_pos = path.clip_to_bbox([0,-1,2,2], inside=True)
patch_pos = matplotlib.patches.PathPatch(path_pos, facecolor='yellowgreen', lw=2, ec='#4c4c4c')

ax.add_patch(patch_neg)
ax.add_patch(patch_pos)

ax.set_xlabel('sentiment score', fontproperties=ax_prop, color='#555555')
ax.set_title("\n".join(wrap(sentiment_str,15)),fontproperties=prop,x=-0.4,y=0.2, color='#555555')
plt.figtext(.5,.92, 'Sentiment of tweets from @' + screen_name + ' - ' + str(datetime.now().date()), fontproperties=ax_prop, fontsize=8, ha='center')

plt.xticks(fontproperties=tick_prop,color='#555555')
plt.tight_layout()
plt.savefig('sentiment_plt.png')

#sentiment_transparent = Image.new('RGBA', (1024, 512), (255,255,255,254))
#sentiment_white = Image.new('RGBA', (1022, 512), (255,255,255,255))
#sentiment_img = Image.open('sentiment_plt.png')

#new_box = (0,0, sentiment_transparent.size[0]-2,sentiment_transparent.size[1])
#sentiment_transparent.paste(sentiment_white,new_box)

#x_margin = sentiment_transparent.size[0] - sentiment_img.size[0]
#y_margin = sentiment_transparent.size[1] - sentiment_img.size[1]

#new_box = (x_margin-int(x_margin/2),y_margin-int(y_margin/2), sentiment_transparent.size[0]-int(x_margin/2),sentiment_transparent.size[1]-int(y_margin/2))
#sentiment_transparent.paste(sentiment_img,new_box)
#sentiment_transparent.save('sentiment_plt.png')

print('Create wordcloud...')

######WordCloud
def clamp(color):
    return min(255,max(0,color))

max_color = 188
pos_r = 154
pos_g = 205
pos_b = 50
def pos_func(word, font_size, position, orientation,
                    **kwargs):
    max_sent = max(pos_sent.values())
    min_sent = min(pos_sent.values())
    grad = 255-(((pos_sent[word]-min_sent)/(max_sent-min_sent))*255)
    return '#%02x%02x%02x' % (clamp(int((pos_r+grad)/2)), clamp(int((pos_g+grad)/2)), clamp(int((pos_b+grad)/2)))

custom_stopwords = {screen_name,'RT','will'}
wordcloud = WordCloud(width=696,height=524,font_path='Libre_Franklin/LibreFranklin-SemiBold.ttf',stopwords=STOPWORDS.union(custom_stopwords),background_color='white',max_font_size=128).generate(pos_text)
plt.figure(figsize=(8.7,6.55))
plt.axis("off")
fig = plt.gcf()
wordcloudname = 'wordcloud.png'
fig.savefig(wordcloudname, bbox_inches='tight')

neg_r = 240
neg_g = 128
neg_b = 128
def neg_func(word, font_size, position, orientation,
                    **kwargs):
    max_sent = -(min(neg_sent.values()))
    min_sent = -(max(neg_sent.values()))
    grad = 255-(((-neg_sent[word]-min_sent)/(max_sent-min_sent))*255)
    return '#%02x%02x%02x' % (clamp(int((neg_r+grad)/2)), clamp(int((neg_g+grad)/2)), clamp(int((neg_b+grad)/2)))

plt.figure(figsize=(8.7,6.55))
plt.axis("off")
wordcloud_neg = WordCloud(width=696,height=524,font_path='Libre_Franklin/LibreFranklin-SemiBold.ttf',stopwords=STOPWORDS.union(custom_stopwords),background_color='white',max_font_size=128).generate(neg_text)
fig = plt.gcf()
wordcloudname = 'wordcloud_neg.png'
fig.savefig(wordcloudname, bbox_inches='tight')

pos_cloud = Image.open('wordcloud.png')
pos_x = pos_cloud.size[0]
pos_y = pos_cloud.size[1]
neg_cloud = Image.open('wordcloud_neg.png')
neg_x = neg_cloud.size[0]
neg_y = neg_cloud.size[1]

img = Image.new('RGB', (pos_x, 80), color = (255, 255, 255))

fnt = ImageFont.truetype('Libre_Franklin/LibreFranklin-SemiBold.ttf', 50)
d = ImageDraw.Draw(img)
d.text((10,10), "       Positive tweets", font=fnt, fill='#555555')
img.save('positive_tweets_text.png')

img2 = Image.new('RGB', (neg_x, 80), color = (255, 255, 255))
d2 = ImageDraw.Draw(img2)
d2.text((10,10), "    Negative tweets", font=fnt, fill='#555555')
img2.save('negative_tweets_text.png')

fnt_small = ImageFont.truetype('Libre_Franklin/LibreFranklin-Thin.ttf', 16)
img3 = Image.new('RGB', (pos_x + neg_x, 41), color = (255, 255, 255))
d3 = ImageDraw.Draw(img3)
d3.text((10,10), "Tweets from @" + screen_name + ' - ' + str(datetime.now().date()), font=fnt_small, fill='#657786')
img3.save('source.png')

h1 = Image.open('positive_tweets_text.png')
h1_x = h1.size[0]
h1_y = h1.size[1]

h2 = Image.open('negative_tweets_text.png')
h2_x = h2.size[0]
h2_y = h2.size[1]
source = Image.open('source.png')
source_x = source.size[0]
source_y = source.size[1]

x = h1_x + h2_x
y = h1_y + pos_y + source_y

#add h2
crop = (0, 0, x, h1_y)
new_box = (h1_x, 0, x, h1_y)
h1 = h1.crop(crop)
#paste second header
h1.paste(h2,new_box)

#add images
crop = (0, 0, x, y)
new_box = (0, h1_y, pos_x, h1_y + pos_y)
h1 = h1.crop(crop)
h1.paste(pos_cloud,new_box)

new_box = (pos_x, h2_y, x, h1_y + pos_y)
h1.paste(neg_cloud,new_box)

#add source
new_box = (0, h2_y + pos_y, x, y)
h1.paste(source,new_box)

plt.axis('off')
h1.save('wordcloud_plt.png')

#transparent1 = Image.new('RGBA', (1024, 512), (255,255,255,254))
#x_margin = int((1024-h1.size[0])/2)
#y_margin = int((512-h1.size[1])/2)
#new_box = (2,0, h1.size[0]+2,h1.size[1])
#transparent1.paste(h1,new_box)
#transparent1.save('wordcloud_plt.png')

print('Create approval plot...')

#rcParams['axes.titlepad'] = 30 

#approval_prop = fm.FontProperties(fname='Libre_Franklin/LibreFranklin-SemiBold.ttf',size=28)

#num_days = 30
#data = approval_estimates.head(num_days)
#plt.figure(figsize=(17,6.5))
#plt.plot_date(data.modeldate, data.approve_estimate,ls='solid',c='lightcoral')
#plt.plot_date(data.modeldate, data.disapprove_estimate,ls='solid',c='yellowgreen')
#plt.title(approval_str, fontproperties=approval_prop,color='#555555')
#plt.figtext(.81,-.01, 'Source: FiveThirtyEight - ' + str(datetime.now().date()), fontproperties=ax_prop, fontsize=12, ha='center')
#plt.xticks(fontproperties=tick_prop,color='#555555')
#plt.yticks(fontproperties=tick_prop,color='#555555')
#plt.savefig('approval_plt.png', bbox_inches='tight')

#approval_transparent = Image.new('RGBA', (1024, 512), (255,255,255,254))
#approval_white = Image.new('RGBA', (1022, 512), (255,255,255,255))
#approval_img = Image.open('approval_plt.png')

#new_box = (0,0, approval_transparent.size[0]-2,approval_transparent.size[1])
#approval_transparent.paste(approval_white,new_box)

#x_margin = approval_transparent.size[0] - approval_img.size[0]
#y_margin = approval_transparent.size[1] - approval_img.size[1]

#new_box = (x_margin-int(x_margin/2),y_margin-int(y_margin/2), approval_transparent.size[0]-int(x_margin/2),approval_transparent.size[1]-int(y_margin/2))
#approval_transparent.paste(approval_img,new_box)
#approval_transparent.save('approval_plt.png')

print('Upload images, get media_ids, and tweet...')

# upload images and get media_ids
#filenames = ['wordcloud_plt.png', 'stats_plt.png', 'sentiment_plt.png', 'approval_plt.png']
#media_ids = []
#for filename in filenames:
#    res = api.media_upload(filename)
#    media_ids.append(res.media_id)

# tweet with multiple images
#api.update_status(status=final_str, media_ids=media_ids)
#print('Tweeted successfully!')