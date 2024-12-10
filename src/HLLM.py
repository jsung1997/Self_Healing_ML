import ast

from pydantic import BaseModel, PrivateAttr
from typing import Any, Dict, Optional
import pandas as pd
from openai import AzureOpenAI
import warnings
import hashlib
import re
import itertools
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import ast
import numpy as np

class H_LLM(BaseModel):
    llm: AzureOpenAI
    config: dict
    context: str = None
    verbose: bool = False
    n: int = 10
    issues: Optional[str] = None
    covariate_guesses: Optional[str] = None
    information: Dict[str, Any] = {}
    information_history: Dict[str, Any] = {}
    datasets: Dict[str, Any] = {}
    _solution_cache: Dict[str, Dict] = PrivateAttr(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def calculate_covariate_performance(self, x, y, model):
        """
        Calculate the model performance (accuracy) across each covariate.
        """
        performance_metrics = {}

        for col in x.columns:
            performance_metrics[col] = {}
            # Discretize the covariate into bins and get bin edges
            x_col, bin_edges = pd.cut(x[col], bins=10, retbins=True, labels=False)
            x['bin'] = x_col

            for bin_val in x['bin'].unique():
                if pd.notna(bin_val):
                    x_bin = x[x['bin'] == bin_val]
                    y_bin = y[x_bin.index]

                    if len(x_bin) > 0:
                        preds = model.predict(x_bin.drop(columns=['bin']))
                        accuracy = accuracy_score(y_bin, preds)
                        # Use bin ranges as keys in the performance_metrics dictionary
                        bin_range = f"{bin_edges[bin_val]:.2f}-{bin_edges[bin_val+1]:.2f}"
                        performance_metrics[col][bin_range] = accuracy

            x.drop(columns=['bin'], inplace=True)

        output_str = f"""The following is accuracy for different ranges of each variable (however, the variables might be continuous form in the original dataset): {performance_metrics}"""
        return output_str

    def hypothesize_issues_with_performance(self, x_before, x_after, y_before, y_after, model, context):
        """
        Hypothesizes 10 possible issues based on the differences between x_before and x_after,
        considering the context and user-specified information, along with the performance of the model
        across each covariate.
        """
        # Discretize covariates and calculate performance metrics
        covariate_performance_before = self.calculate_covariate_performance(x_before, y_before, model)
        covariate_performance_after = self.calculate_covariate_performance(x_after, y_after, model)

        task = f"""
        Given the following information:
        - Data before the shift: {x_before.describe()}
        - Data after the shift: {x_after.describe()}
        - Context: {context}
        - Model performance across each covariate before the shift: {covariate_performance_before}
        - Model performance across each covariate after the shift: {covariate_performance_after}

        You know for a fact that the model has degraded. Analyze the covariates and think why.
        
        Review each existing covariate and provide a hypothesis on whether it might have changed and resulted in the model underperforming. Provide evidence for each hypothesis and the strength of belief for each covariate.

        Format your output as follows:
        Covariate: <covariate>; Hypothesis: ...; Evidence: ...; Strength of belief: ...

        After reviewing all the covariates, assign a confidence score for each covariate indicating your confidence level that the covariate has issues. Use the following confidence levels: extremely confident, confident, somewhat confident, unsure, completely unsure. Only use 'extremely confident' if you have overwhelming evidence for your decision. Prioritize making more confident beliefs. Avoid being uncertain. Use the available inputs as well as the data to make the best possible decision. Your goal is to be correct while reducing entropy of the probabilities (be confidently correct).
        """

        covariate_guesses = self._get_llm_response(task)
        self.covariate_guesses = covariate_guesses
        second_query = f"""This was the previous task: {task} and your response: {covariate_guesses}. TASK: Re-rank each covariate based on the given data. You should only rank "confident" or above if you are absolutely certain there is an issue. Substantially decrease your confidence for hypotheses that are not supported and Substantially increase your confidence for hypotheses that are supported. Avoid false positives. Prioritize making more confident beliefs. Avoid being uncertain. Use the available inputs as well as the data to make the best possible decision. Your goal is to be correct WHILE REDUCING THE ENTROPY of the probabilities (be confidently correct).  """
        second_response = self._get_llm_response(second_query)

        return second_response
    
    def hypothesize_issues_with_performance(self, x_before, x_after, y_before, y_after, model, context):
        """
        Hypothesizes 10 possible issues based on the differences between x_before and x_after,
        considering the context and user-specified information, along with the performance of the model
        across each covariate.
        """
        # Discretize covariates and calculate performance metrics
        covariate_performance_before = self.calculate_covariate_performance(x_before, y_before, model)
        covariate_performance_after = self.calculate_covariate_performance(x_after, y_after, model)

        task = f"""
        Given the following information:
        - Data before the shift: {x_before.describe()}
        - Data after the shift: {x_after.describe()}
        - Context: {context}
        - Model performance across each covariate before the shift: {covariate_performance_before}
        - Model performance across each covariate after the shift: {covariate_performance_after}

        You know for a fact that the model has degraded. Analyze the covariates and think why.
        
        Review each existing covariate and provide a hypothesis on whether it might have changed and resulted in the model underperforming. Provide evidence for each hypothesis and the strength of belief for each covariate.

        Format your output as follows:
        Covariate: <covariate>; Hypothesis: ...; Evidence: ...; Strength of belief: ...

        After reviewing all the covariates, assign a confidence score for each covariate indicating your confidence level that the covariate has issues. Use the following confidence levels: extremely confident, confident, somewhat confident, unsure, completely unsure. Only use 'extremely confident' if you have overwhelming evidence for your decision. Prioritize making more confident beliefs. Avoid being uncertain. Use the available inputs as well as the data to make the best possible decision. Your goal is to be correct while reducing entropy of the probabilities (be confidently correct).
        """

        covariate_guesses = self._get_llm_response(task)
        self.covariate_guesses = covariate_guesses
        second_query = f"""This was the previous task: {task} and your response: {covariate_guesses}. TASK: Re-rank each covariate based on the given data. You should only rank "confident" or above if you are absolutely certain there is an issue. Substantially decrease your confidence for hypotheses that are not supported and Substantially increase your confidence for hypotheses that are supported. Avoid false positives. Prioritize making more confident beliefs. Avoid being uncertain. Use the available inputs as well as the data to make the best possible decision. Your goal is to be correct WHILE REDUCING THE ENTROPY of the probabilities (be confidently correct).  """
        second_response = self._get_llm_response(second_query)

        return second_response
    
    def hypothesize_issues_with_performance_covariate_combinations(self, x_before, x_after, y_before, y_after, model, context, n=20):
        """
        Hypothesizes 10 possible issues based on the differences between x_before and x_after,
        considering the context and user-specified information, along with the performance of the model
        across each covariate.
        """
        # Discretize covariates and calculate performance metrics
        covariate_performance_before = self.calculate_covariate_performance(x_before, y_before, model)
        covariate_performance_after = self.calculate_covariate_performance(x_after, y_after, model)

        task = f"""
        Given the following information:
        - Data before the shift: {x_before.describe()}
        - Data after the shift: {x_after.describe()}
        - Context: {context}
        - Model performance across each covariate before the shift: {covariate_performance_before}
        - Model performance across each covariate after the shift: {covariate_performance_after}

        You know for a fact that the model has degraded. Analyze the covariates and think why.
        
        Then, hypothesize {n} possible covariates or combinations of covariates that might have changed and resulted in the model underperforming. Each possibility should be mutually exclusive. For example, [X1] is one possibility, [X2] is another, and [X1, X2] is a third.
        """

        covariate_guesses = self._get_llm_response(task)
        self.covariate_guesses = covariate_guesses
        second_query = f"""This was the previous task: {task} and your response: {covariate_guesses}. TASK: re-rank each hypothesis of covariates based on which policy is likely to be correct, prioritizing most likely ones first. Format your output as a list of lists, such as: [[x1], [x1, x2], ...]"""
        second_response = self._get_llm_response(second_query)
        third_query = f"""This was our previous conversation and your response: INPUT: {second_query}. OUTPUT: {second_response}. TASK: Use this list to create specific query conditions of the people that should be removed using these covariates from the dataset. The removal should follow a dataframe query structure. Format each query as follows: Query 1: <>; Query 2: <>; ..."""
        third_response = self._get_llm_response(third_query)
        return second_response, third_response
    
    def summarize_probabilities(self, covariate_guesses, x_before, x_after, context):
        """
        Summarizes the initial hypotheses and assigns calibrated probabilities that sum to 100%.
        """
        task = f"""
        Given the following information:
        - Data before the shift: {x_before.describe()}
        - Data after the shift: {x_after.describe()}
        - Context: {context}
        - Initial hypotheses on covariates or combinations of covariates that might have changed and resulted in model underperformance: {covariate_guesses}

        Summarize the provided hypotheses and assign probabilities to each hypothesis such that the total probability sums to 100%. 

        Your probabilities should be reflective of the evidence and data. Uniform probabilities (10% each) implies no knowledge. 100% probability on one covariate implies certain belief. Prioritize making more confident beliefs. Avoid being uncertain. Use the available inputs as well as the data to make the best possible decision. Your goal is to be correct while reducing entropy of the probabilities (be confidently correct).

        Format each hypothesis and its probability as follows:
        Hypothesis: [<covariate1>, <covariate2>, ...]; Probability: <probability>
        """

        final_hypotheses = self._get_llm_response(task)
        return final_hypotheses

    
    def hypothesize_issues(self, x_before, x_after, context):
        """
        Hypothesizes 10 possible issues based on the differences between x_before and x_after,
        considering the context and user-specified information.
        """
        task = f"""
        Given the following information:
        - Data before the shift: {x_before.describe()}
        - Data after the shift: {x_after.describe()}
        - Context: {context}
        
        Hypothesize {self.n} possible issues in the new dataset that might have resulted in the model underperforming. Provide the possible issue, evidence in the data, and confidence (1-10). Confidence 10 means you are very certain this issue is correct. Confidence 1 is that you are not certain and this is a guess.

        Format of the output: Issue: <>; Evidence: <>; Confidence: <> 

        """
        
        issues = self._get_llm_response(task)
        self.issues = issues
        return issues
    
    def _generate_cache_key(self, x1: pd.DataFrame, x2: pd.DataFrame) -> str:
        """
        Generates a unique cache key based on the DataFrame columns.
        """
        column_string = ','.join(sorted(x1.columns))
        return hashlib.sha256(column_string.encode()).hexdigest()
    
    def clear_cache(self):
        """Clears the subgroup cache."""
        self._solution_cache.clear()
        if self.verbose:
            print("Cache cleared.")
    
    def suggest_solutions_retrain_model(self, issues, x_before, x_after):
        """Suggest solutions based on removing data"""
        task = f"""
        Suppose the following issues in the dataset: {issues}
        Data before the shift: {x_before.describe()}
        Data after the shift: {x_after.describe()}

        Suggest {self.n} possible subgroups that might need re-training. That is, fitting a separate model on these subgroups might result in superior performance. 
        The subgroups can be single (e.g. X > x) but could also be multiple combinations (e.g. X > x and Y < y)
        """
        
        solution = self._get_llm_response(task)
        
        return solution

    def suggest_solutions_remove_data(self, issues, x_before, x_after):
        """
        Suggests solutions for each identified issue in the form of pandas filter queries 
        to remove corrupted data from x_after.
        """
        task = f"""
        Suppose the following issues in the dataset: {issues}
        Data before the shift: {x_before.describe()}
        Data after the shift: {x_after.describe()}

        Suggest {self.n} possible subgroups that if removed could result in better performance for the model.
        The subgroups can be single (e.g. X > x) but could also be multiple combinations (e.g. X > x and Y < y)
        """
        
        solution = self._get_llm_response(task)
        
        return solution
    
    def convert_to_list_of_queries(self, solutions, x_before):

        task = f"""
        Suppose the following dataframe queries: {solutions}. TASK: Return these queries such that they meet two conditions: (1) you return ONLY a list; (2) each item in a list is formatted in the a pandas query format that can be used to filter datasets. The following are the available columns: {x_before.columns}.
        Return ONLY what is required. EXAMPLE: ['X > 80', 'Y < 20'].
        """
                
        list_formatted = self._get_llm_response(task)
        
        return ast.literal_eval(list_formatted)

    def get_queries(self, x_before, x_after, all_queries=False):
        """
        Runs the H-LLM system to hypothesize issues and suggest solutions.
        Returns a list of 10 pandas queries as solutions.
        """
        # Store information history if dictionary is not empty
        if self.information:
            self.information_history[len(self.information_history)] = self.information
        context = self.context
        # Use a cache key
        cache_key = self._generate_cache_key(x_before, x_after)
        if cache_key in self._solution_cache:
            return self._solution_cache[cache_key]['queries_remove']


        issues = self.hypothesize_issues(x_before, x_after, context)

        solutions = self.suggest_solutions_remove_data(issues, x_before, x_after)
        queries_remove = self.convert_to_list_of_queries(solutions, x_before)

        if all_queries:
            solutions_retrain = self.suggest_solutions_retrain_model(issues, x_before, x_after)
            queries_retrain = self.convert_to_list_of_queries(solutions_retrain, x_before)

            return queries_remove, queries_retrain
        
        # Store cache
        self._solution_cache[cache_key] = {'queries_remove': queries_remove}
        self.information['solutions'] = solutions
        self.information['issues'] = issues
        self.information['queries_remove'] = queries_remove
        return queries_remove

    def fit_many_policies(self, model, x_before, x_after, y_after, x_backtest, y_backtest):
        """Fit and return many policies"""
        context = self.context
        # Use a cache key
        cache_key = self._generate_cache_key(x_before, x_after)
        if cache_key in self._solution_cache:
            return self._solution_cache[cache_key]

        queries_remove, queries_retrain = self.get_queries(x_before, x_after, all_queries=True)
        queries_leave = ['not (' + x + ')' for x in queries_remove]
        
        # Evaluate remove queries on backtesting window
        scores_remove = []
        for cond in queries_leave:
            xi_filt = x_after.query(cond)
            yi_filt = y_after[xi_filt.index]
            
            try:
                model.fit(xi_filt, yi_filt)
                preds = model.predict(x_backtest)
                score = accuracy_score(y_backtest, preds)
                scores_remove.append(score)
            except:
                scores_remove.append(0)  # Append 0 in case of an error, e.g., insufficient data for training

        # Evaluate retrain queries on backtesting window
        scores_retrain = []
        for cond in queries_retrain:
            xi_filt = x_after.query(cond)
            yi_filt = y_after[xi_filt.index]

            try:
                model.fit(xi_filt, yi_filt)
                preds = model.predict(x_backtest)
                score = accuracy_score(y_backtest, preds)
                scores_retrain.append(score)
            except:
                scores_retrain.append(0)  # Similar error handling

        # Return all queries and their corresponding scores
        self._solution_cache[cache_key] = {'queries_remove': queries_remove, 'queries_retrain': queries_retrain, 'scores_remove': scores_remove, 'scores_retrain': scores_retrain}

        return {'queries_remove': queries_remove, 'queries_retrain': queries_retrain, 'scores_remove': scores_remove, 'scores_retrain': scores_retrain}


    def fit_model_no_testing(self, model, x_before, x_after, y_after, x_backtest, y_backtest):
        """Fit the model based on the optimal policy and adapt"""
        # Get which queries to leave
        queries_remove = self.get_queries(x_before, x_after)
        queries_leave = ['not (' + x + ')' for x in queries_remove]

        query_leave = queries_leave[0]
       
        x_filtered = x_after.query(query_leave)
        y_filtered = y_after[x_filtered.index]

        model.fit(x_filtered, y_filtered)
        return model

    def fit_model(self, model, x_before, x_after, y_after, x_backtest, y_backtest):
        """Fit the model based on the optimal policy and adapt"""
        # Get which queries to leave
        queries_remove = self.get_queries(x_before, x_after)
        queries_leave = ['not (' + x + ')' for x in queries_remove]

        # Evaluate on the backtesting window
        scores = []
        for cond in queries_leave:
            xi_filt = x_after.query(cond)
            yi_filt = y_after[xi_filt.index]
            
            try:
                model.fit(xi_filt, yi_filt)
                preds = model.predict(x_backtest)
                score = accuracy_score(y_backtest, preds)
                scores.append(score)
            except:
                scores.append(0)

        query_leave = queries_leave[np.argmax(scores)]

        x_filtered = x_after.query(query_leave)
        y_filtered = y_after[x_filtered.index]

        self.information['query_leave'] = query_leave
        self.information['scores'] = {query: score for query, score in zip(queries_leave, scores)}
        
        self.datasets['x_filtered'] = x_filtered
        self.datasets['y_filtered'] = y_filtered
        model.fit(x_filtered, y_filtered)
        return model

    def fit_model_no_diagnosis(self, model, x_before, x_after, y_after, x_backtest, y_backtest):
        """Fit the model based on the optimal policy and adapt"""
        # Get which queries to leave
        issues = """Unknown."""
        solutions = self.suggest_solutions_remove_data(issues, x_before, x_after)
        queries_remove = self.convert_to_list_of_queries(solutions, x_before)
        queries_leave = ['not (' + x + ')' for x in queries_remove]

        print(queries_leave)
        # Evaluate on the backtesting window
        scores = []
        for cond in queries_leave:
            xi_filt = x_after.query(cond)
            yi_filt = y_after[xi_filt.index]
            
            try:
                model.fit(xi_filt, yi_filt)
                preds = model.predict(x_backtest)
                score = accuracy_score(y_backtest, preds)
                scores.append(score)
            except:
                scores.append(0)

        query_leave = queries_leave[np.argmax(scores)]
        print(query_leave)

        x_filtered = x_after.query(query_leave)
        y_filtered = y_after[x_filtered.index]


        model.fit(x_filtered, y_filtered)
        return model

        
    def _get_llm_response(self, input_text, system_message=None, metadata_output=False, modelid=None):
            if self.verbose:
                print('----------INPUT TEXT --------------')
                print(input_text)

            if system_message is None:
                # LLM response with/without a system message
                response = self.llm.chat.completions.create(
                    model=self.config["engine"] if modelid is None else self.conversion_dict[modelid],
                    messages=[{"role": "user", "content": input_text}],
                    temperature=self.config['temperature'],
                    seed=self.config['seed'],
            )
            else:
                # Get the response from the LLM with a system message
                response = self.llm.chat.completions.create(
                model=self.config["engine"] if modelid is None else self.conversion_dict[modelid],
                    messages=[{"role": "system", "content": system_message}, 
                                {"role": "user", "content": input_text}],
                    temperature=self.config['temperature'],
                    seed=self.config['seed'],
            )
            message = response.choices[0].message.content

            if self.verbose:
                print('----------LLM RESPONSE TEXT--------------')
                print(message)

            if metadata_output:
                metadata = {'tools': response.choices[0].message.tool_calls,
                            'function calls': response.choices[0].message.function_call}
                return message, metadata     
            else:

                return message
            
    def extract_issues_evidence_confidence(self):
        """
        Extracts issues, evidence, and confidence from the provided text and puts them into a pandas dataframe.
        Each issue, evidence, and confidence are in separate columns.
        The number of rows is equal to the number of issues.

        Parameters:
        issues (str): A string containing issues, evidence, and confidence.

        Returns:
        pd.DataFrame: A dataframe with issues, evidence, and confidence.
        """
        # Split the issues string into individual issue entries
        issue_entries = self.issues.split('\n')

        # Lists to store issues, evidence, and confidence
        issue_list = []
        evidence_list = []
        confidence_list = []

        # Process each issue entry to extract issue, evidence, and confidence
        for entry in issue_entries:
            if entry.strip():  # Skip empty entries
                # Split the entry by semicolon to separate issue, evidence, and confidence
                parts = entry.split(';')
                if len(parts) == 3:
                    issue, evidence, confidence = [part.strip() for part in parts]
                    issue_list.append(issue)
                    evidence_list.append(evidence)
                    confidence_list.append(confidence)

        # Create a DataFrame with issues, evidence, and confidence
        df = pd.DataFrame({
            'Issue': issue_list,
            'Evidence': evidence_list,
            'Confidence': confidence_list
        })

        return df