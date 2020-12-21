## import from the functions document:
from functions import * 

## merge submissions
merge_submissions() 

## merge comments 
merge_comments() 

## clean bots 
## use manual = True the first time. 
bot_clean(manual = False, done = True) 

## concatenate data
concat_data(done = True) ## important to keep this name

## smaller cleaning tasks
small_cleaning(done = True)

## more cleaning tasks 
clean_comment(done = True)

## specific cleaning for different purposes: 
#  get accepts: ['thread', 'tree'] as arguments. 
H1_preprocess(get = ['submission'], done = False) 

## you are done! :) 