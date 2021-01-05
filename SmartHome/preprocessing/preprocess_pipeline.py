## import from the functions document:
from functions import * 

## merge submissions
merge_submissions() 

## merge comments 
merge_comments() 

## clean bots 
## use manual = True the first time. 
bot_clean(manual = True, done = False) 

## concatenate data
concat_data(done = False) ## important to keep this name

## smaller cleaning tasks
small_cleaning(done = False)

## more cleaning tasks 
clean_comment(done = False)

## specific cleaning for different purposes: 
#  get accepts: ['thread', 'tree'] as arguments. 
gen_subsets(get = ['submission', "tree", "thread"], done = False) 

## you are done! :) 