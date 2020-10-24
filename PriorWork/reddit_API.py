import praw

## updated to be Victor's Reddit account.
user_agent = ("Smart Home v1.0 by /u/NLPinfiltrator")

reddit = praw.Reddit(client_id='fQuRCESlv_7yOQ',
                     client_secret='v6AjzytAKDgqjqyQNczapezXUjk',
                     username='NLPinfiltrator',
                     password='BestPass0701',
                     user_agent=user_agent)

subreddit = reddit.subreddit('smarthome')

# Init a dictionary where to store all info
# {post_id: [parent_comment, {reply_id:[votes, reply_content]}}
conversedict = {}
hot_smarthome = subreddit.hot(limit=10)

for submission in hot_smarthome:
    # boolean stickied/pinned post
    if not submission.stickied:
        print('Title: {}, ups: {}, downs: {}'.format(submission.title,
                                                     submission.ups,
                                                     submission.downs))
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
            # all comments parent and child
            conversedict[comment.id] = [
                comment.body.encode('utf-8', errors='ignore'), {}]
            # if comment parent is not refering to the post id >> is not a first tier comment
            # the comment needs to populate the inner dict
            if comment.parent() != submission.id:
                parent = str(comment.parent())
                conversedict[parent][1][comment.id] = [
                    comment.ups, comment.body.encode('utf-8', errors='ignore')]


for id_, body in conversedict.items():
    print('\n', id_)
    print('\n', body)
    print('-'*20)

# homeautomation
# smarthomeautomation
