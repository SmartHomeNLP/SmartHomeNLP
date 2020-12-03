# import wordclouds
from wordcloud import WordCloud

# initiate wordcloud object
def clouds(num_topics): 
    wc = WordCloud(background_color="white", colormap="Dark2", max_font_size=150, random_state=42)

    # set the figure size
    plt.rcParams['figure.figsize'] = [20, 15]

    # Create subplots for each topic
    for i in range(num_topics):

        wc.generate(text = topics_df["Terms per Topic"][i])
        
        plt.subplot(5, 4, i+1)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(topics_df.index[i])

    plt.show()