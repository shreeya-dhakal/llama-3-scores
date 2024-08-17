import streamlit as st

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class LLaMAScoreAnalyzer:
    def __init__(self):
        self.languages = ["Nepali", "Hindi"]
        self.models = ["Baseline", "LoRA"]
        self.scores_gpt = ["relevance_score", "cc_score", "syntax_score", "complete_score"]
        self.rouge_bleu = ["rougeL", "bleu"]
        self.categories = ["hallucination_type", "is_repeat"]
        self.DATA_PATH = {
                "Nepali": {"Baseline": "./data/nepali_baseline_all_scores.csv", "LoRA": "./data/nepali_lora_all_scores.csv"},
                "Hindi": {"Baseline": "./data/hindi_baseline_all_scores.csv", "LoRA": "./data/nepali_baseline_all_scores.csv"}
            }
        
    def load_samples(self, lang):
        cols_to_show = ["instruction", "input", "output"]
        for model in self.DATA_PATH[lang]:
            df = pd.read_csv(self.DATA_PATH[lang][model])
            df.rename({"output": "expected_output"})
            df[model+"_Response"] = df["cleaned_response"]
            cols_to_show.append(model+"_Response")
        cols_to_show = cols_to_show + ["relevance_score", "cc_score", "syntax_score", "complete_score", "rougeL", "blue", "is_repeat", "hallucination_type"]
        df = df[[col for col in cols_to_show if col in df.columns]]
        st.write(df.sample(5))
        

    def load_data(self, lang, model):
        df = pd.read_csv(self.DATA_PATH[lang][model])
        df['Language'] = lang
        df['Model'] = model
        return df

    def draw_specific_plots(self, data, categories, x_variable, title):
        fig, ax = plt.subplots(figsize=(12, 6))

        palette = sns.color_palette("pastel", len(categories) * len(data[x_variable].unique()))

        for i, category in enumerate(categories):
            for j, unique_value in enumerate(data[x_variable].unique()):
                subset = data[data[x_variable] == unique_value]
                sns.kdeplot(data=subset, x=category, fill=True, common_norm=False, alpha=0.5,
                            ax=ax, color=palette[i * len(data[x_variable].unique()) + j],
                            label=f"{category} ({unique_value})")
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(f"{x_variable}", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(title="Category (Language/Model)")
        
        return fig

    def draw_combined_density_plot(self, data, title):
        fig, ax = plt.subplots(figsize=(12, 8))

        palette = sns.color_palette("pastel", len(self.scores_gpt))

        for i, category in enumerate(self.scores_gpt):
            sns.kdeplot(data=data, x=category, fill=True, common_norm=False, alpha=0.5, ax=ax, label=category, color=palette[i])
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Score", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(title="Score Categories")

        return fig
    
    def draw_bar_plot(self, data, categories, x_variable, title):
        fig, axs = plt.subplots(len(categories), 1, figsize=(10, 6 * len(categories)))

        palette = sns.color_palette("pastel", len(categories))

        if len(categories) == 1:
            axs = [axs]  # Ensure axs is iterable even for a single plot

        for i, category in enumerate(categories):
            sns.countplot(data=data, x=category, hue=x_variable, palette=palette, ax=axs[i])
            axs[i].set_title(f"Distribution of {category} for {title}", fontsize=16)
            axs[i].set_xlabel(category, fontsize=12)
            axs[i].set_ylabel("Count", fontsize=12)
            axs[i].legend(title=x_variable)

        plt.tight_layout()
        return fig
    
    def score_analyzer(self):
        st.sidebar.markdown("""
                    This App was created as a part of the project: "Fine-tuning LLaMA 3 with Low-Rank Adaptation for Nepali and Hindi"
                            """)   
        st.title("Findings from Fine-tuning LLaMA 3 with Low-Rank Adaptation for Nepali and Hind! ")
        st.markdown("""
                    Full post here: 
            """)
        show_samples = st.sidebar.checkbox("Show Sample Data", value=False)
        detailed_view = st.sidebar.checkbox("Enable Detailed Charts View", value=False)

        selected_languages = st.sidebar.multiselect("Select Languages", self.languages, default="Nepali")
        selected_gpt_scoring = st.sidebar.multiselect("Select Score Category", self.scores_gpt, default="relevance_score")
        selected_models = st.sidebar.multiselect("Select Models", self.models, default="Baseline")

        dfs = []
        for lang in selected_languages:
            for model in selected_models:
                df = self.load_data(lang, model)
                dfs.append(df)
        if show_samples:
            for lang in selected_languages:
                st.write(f"Sample data for {lang}")
                self.load_samples(lang)
        
        combined_data = pd.concat(dfs, ignore_index=True)
        if detailed_view:
            for language in selected_languages:
                language_data = combined_data[combined_data['Language'] == language]
                title = f"Distribution of {selected_gpt_scoring}  for {language}"
                fig = self.draw_specific_plots(language_data, selected_gpt_scoring, 'Model', title)
                st.pyplot(fig)
            if len(selected_languages) > 1:
                for model in selected_models:
                    model_data = combined_data[combined_data['Model'] == model]
                    title = f"Distribution of {selected_gpt_scoring} for {model}"
                    fig = self.draw_specific_plots(model_data, selected_gpt_scoring, 'Language', title)
                    st.pyplot(fig)
            
            st.sidebar.markdown("""
                    Show additional evaluation scores and categories below:
                            """)   
            additional_score_categories = st.sidebar.checkbox("Hallucination and Instruction Repeat Statistics", value=False)
            if additional_score_categories:
                additional_categories = st.sidebar.multiselect("Select Category", self.categories, default="hallucination_type")
                for language in selected_languages:
                    language_data = combined_data[combined_data['Language'] == language]
                    title = f"{language}"
                    fig = self.draw_bar_plot(language_data, additional_categories, 'Model', title)
                    st.pyplot(fig)
                if len(selected_languages) > 1:
                    for model in selected_models:
                        model_data = combined_data[combined_data['Model'] == model]
                        title = f"{model}"
                        fig = self.draw_bar_plot(model_data, additional_categories, 'Language', title)
                        st.pyplot(fig)
        else:
           for language in selected_languages:
                for model in selected_models:
                    title = f"Distribution of Scores for Different Evaluation Criterias for {language} [{model} Model]"
                    fig = self.draw_combined_density_plot(combined_data[(combined_data['Language'] == language) & 
                                                                        (combined_data['Model'] == model)], title)
                    st.pyplot(fig)

        


def main():
    
    st.sidebar.header("Findings from Fine-tuning LLaMA 3 with Low-Rank Adaptation for Nepali and Hindi!")

    analyzer = LLaMAScoreAnalyzer()
    analyzer.score_analyzer()


    
if __name__ == "__main__":
    main()

