from inspection_fun import * 

if __name__ == "main":
    input_list = ["H1_thread", "H1_tree", "H2_submissions"]
    settings = "a0.01_b0.1_k"

    for i in input_list:
        print(f"Starting to produce dateframes for: {i}")
        corpus = load_pickle(f"{i}_corpus", "corpus")
        data = load_pickle(f"{i}", "data")
        if i == "H1_thread":
            names = ["H1_thread_models_b0.1_b1", "H1_thread_models_b0.1_a0.01_50-100"]
            for model_name in names:
                models = load_pickle(model_name, "model")
                if model_name == "H1_thread_models_b0.1_b1":
                    n_topic = "30"
                else:
                    n_topic = "100"
                dominant_topic(models[settings+n_topic], corpus, data["org_text"], save_name = f"{i}_{n_topic}") 
        
        if i == "H1_tree":
            models = load_pickle(i+"_models", "model")
            n_topics = ["30", "100"]
            for n in n_topics:
                dominant_topic(models[settings+n], corpus, data["org_text"], save_name=f"{i}_{n}")

        if i == "H2_submissions":
            models = load_pickle(i+"_b0.1_a0.01", "model")
            n_topics = ["30", "100"]
            for n in n_topics:
                dominant_topic(models[settings+n], corpus, data["org_text"], save_name = f"{i}_{n}")
