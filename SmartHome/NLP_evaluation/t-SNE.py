
####### T-SNE #######
# https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/?fbclid=IwAR3xeAQMyimOPmtf9Jbk7agWqURyuWsvgMhIjVyh9ZnJtL20mA8vnefEVlk
# Get topic weights and dominant topics ------------
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
import matplotlib.colors as mcolors

# Get topic weights
topic_weights = []
for i, row_list in enumerate(tree[tree_corpus]):
    topic_weights.append([w for i, w in row_list[0]])

# Array of topic weights    
arr = pd.DataFrame(topic_weights).fillna(0).values

# Keep the well separated points (optional)
arr = arr[np.amax(arr, axis=1) > 0.35]

# Dominant topic number in each doc
topic_num = np.argmax(arr, axis=1)

# tSNE Dimension Reduction
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
tsne_lda = tsne_model.fit_transform(arr)

# Plot the Topic Clusters using Bokeh
output_notebook()
n_topics = 17 # was four in theirs, but I think we have 17?
#mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
import random
colorList = np.array(['#' + str(hex(random.randint(0, 16777215)))[2:] for i in range(17)])

plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), 
              plot_width=900, plot_height=700)
plot.scatter(x = tsne_lda[:,0], y = tsne_lda[:,1], color = colorList[topic_num])
show(plot)

####### t-SNE attempt 2 #######
## we have a wrong format..
## chech how it matches doc2bow.
hm = np.array([[y for (x, y, z) in tree[tree_corpus[i]]] for i in range(len(tree_corpus))])
hm

for i in range(len(tree_corpus)): 
    print(i)

(x, y, z) = tree[tree_corpus[196666]] 
x
y
z


