import praw

reddit = praw.Reddit(client_id='V65osQ-P4aMWBQ',
                     client_secret='IeKPqv2TRgpEY2RGo8wf9Uv2iz0',
                     username='mgmtmor',
                     password='sHreddit',
                     user_agent='prawscraper')

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
