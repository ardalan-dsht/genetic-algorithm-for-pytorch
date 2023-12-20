import random
from typing import List

import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class GeneticAlgorithm:
    def __init__(
        self,
        model,
        population_count,
        percentage_of_parents_to_keep,
        cross_over_threshold,
        mutation_change_threshold,
        device,
    ):
        self.model = model
        self.count_of_initial_model_weights = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        self.population_count = population_count
        self.percentage_of_parents_to_keep = percentage_of_parents_to_keep
        model_architecture = {}
        for key_name, weights in model.state_dict().items():
            model_architecture[key_name] = weights.shape
        self.model_architecture = model_architecture
        self.cross_over_threshold = cross_over_threshold
        self.mutation_change_threshold = mutation_change_threshold
        self.device = device
        self.model.to(self.device)
        self.accuracy_during_time_for_generation = []

    def deconstruct_statedict(self, model: torch.nn.Module) -> torch.Tensor:
        """
        Deconstructs the state dictionary of a model into a one-dimensional tensor.

        Args:
            model (torch.nn.Module): The model whose state dictionary is to be deconstructed.

        Returns:
            torch.Tensor: The one-dimensional tensor representation of the model's state dictionary.
        """
        one_dim_statedict = torch.Tensor()
        for key_name, weights in model.state_dict().items():
            flattend_weights = torch.flatten(weights)
            one_dim_statedict = torch.cat((one_dim_statedict, flattend_weights), dim=0)
        return one_dim_statedict

    def reconstruct_statedict(self, flattend_weights: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the state dictionary of a model from a one-dimensional tensor.

        Args:
            flattend_weights (torch.Tensor): The one-dimensional tensor representation of the model's state dictionary.

        Returns:
            torch.Tensor: The reconstructed state dictionary of the model.
        """
        state_dict = {}
        pointer = 0
        flattend_weights = torch.rand(
            self.count_of_initial_model_weights, device=self.device
        )
        for key_name, weights_shape in self.model_architecture.items():
            if len(weights_shape) > 1:
                count_of_weights_this_module_needs = weights_shape[0] * weights_shape[1]
            else:
                count_of_weights_this_module_needs = weights_shape[0]
            slice_of_selected_weights = flattend_weights[
                pointer : pointer + count_of_weights_this_module_needs
            ]

            state_dict[key_name] = torch.reshape(
                slice_of_selected_weights, self.model_architecture[key_name]
            )
            pointer = count_of_weights_this_module_needs + pointer
        return state_dict

    def natural_selection(
        self,
        population: torch.Tensor,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Performs the natural selection process on the population.

        Args:
            population (torch.Tensor): The current population of individuals.
            x_train (torch.Tensor): The training data.
            y_train (torch.Tensor): The training labels.

        Returns:
            List[torch.Tensor]: The best individuals in the population after natural selection.
        """
        x_train = x_train.to(self.device)
        accuracy_for_all_individuals = []
        # Perform forwardpass for each individual
        for idx in range(len(population)):
            state_dict = self.reconstruct_statedict(population[idx])
            self.model.load_state_dict(state_dict)
            y_hat = self.model(x_train)
            y_hat = torch.squeeze(y_hat)
            y_hat = y_hat.to(torch.int16)
            y_hat = y_hat.cpu().numpy()
            accuracy = accuracy_score(y_train, y_hat)
            accuracy_for_all_individuals.append(accuracy)
        # Sort individuals based on accuracy.
        sorted_population = sorted(
            zip(accuracy_for_all_individuals, population), key=lambda x: x[0]
        )
        sorted_population = [t for _, t in sorted_population]
        sorted_population = sorted_population[::-1]
        # Keep only the best individuals.
        threshold = round(len(sorted_population) * self.percentage_of_parents_to_keep)
        best_individuals_in_population = sorted_population[0:threshold]
        accuracy_for_all_individuals.sort()
        self.accuracy_during_time_for_generation.append(accuracy_for_all_individuals[0])
        return best_individuals_in_population

    def cross_over(self, population: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Performs the cross-over operation on the population.

        Args:
            population (List[torch.Tensor]): The current population of individuals.

        Returns:
            List[torch.Tensor]: The new population after the cross-over operation.
        """
        cross_over_idx = round(len(population[0]) * self.cross_over_threshold)
        children = []
        count_of_children_needed = int((self.population_count - len(population)) / 2)
        for idx in range(count_of_children_needed):
            # Find parents
            male = random.sample(population, k=1)[0].to(self.device)
            female = random.sample(population, k=1)[0].to(self.device)
            # Slice genes
            male_first_part = male[0:cross_over_idx].to(self.device)
            male_second_part = male[cross_over_idx::].to(self.device)
            female_first_part = female[0:cross_over_idx].to(self.device)
            felame_second_part = female[cross_over_idx::].to(self.device)
            # Create new children
            child1 = torch.cat((male_first_part, felame_second_part)).to(self.device)
            child2 = torch.cat((female_first_part, male_second_part)).to(self.device)
            children.append(child1)
            children.append(child2)
        return children

    def mutation(self, children: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Performs the mutation operation on the children.

        Args:
            children (List[torch.Tensor]): The children to be mutated.

        Returns:
            List[torch.Tensor]: The mutated children.
        """
        mutated_children = []
        for child in children:
            # Create mutation values.
            # Some random values. => exp. [0.9420, 0.8821, 0.4306, 0.7354, 0.1637]
            mutation_base_values = torch.rand(
                self.count_of_initial_model_weights, device=self.device
            )
            # Scale those random numbers. => exp. [0.0283, 0.0265, 0.0129, 0.0221, 0.0049]
            scaled_mutation_values = (
                mutation_base_values * self.mutation_change_threshold
            ).to(self.device)
            # Get negation signs so weights are gonna increase and decrease. => exp. [ 1,  1,  1, -1, -1]
            negation_signs_for_scaled_mutation_values = (
                torch.randint(
                    0,
                    2,
                    size=(1, self.count_of_initial_model_weights),
                    device=self.device,
                ).squeeze()
                * 2
                - 1
            )
            # Actual values which could be or not added to genes, only added if genes are selected. => exp. [ 0.0019,  0.0040,  0.0018, -0.0296, -0.0187]
            mutation_values_with_negation_signs = torch.mul(
                scaled_mutation_values, negation_signs_for_scaled_mutation_values
            ).to(self.device)
            # Select which genes are gonna be mutated. => exp. [1, 1, 0, 0, 1]
            gene_selection_for_mutation = torch.randint(
                0, 2, (1, self.count_of_initial_model_weights), device=self.device
            ).squeeze()
            # Actual mutation, these values are gonna be added to cross overed children. => exp. [ 0.0211,  0.0058, -0.0000,  0.0000, -0.0172]
            mutation_values = torch.mul(
                gene_selection_for_mutation, mutation_values_with_negation_signs
            ).to(self.device)
            # Perform mutation
            mutated_child = torch.add(child, mutation_values)
            mutated_children.append(mutated_child)
        return mutated_children

    def fit(
        self,
        num_iterations: int,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
    ) -> torch.nn.Module:
        """
        Fits the model using the genetic algorithm.

        Args:
            num_iterations (int): The number of iterations to run the genetic algorithm.
            x_train (torch.Tensor): The training data.
            y_train (torch.Tensor): The training labels.

        Returns:
            torch.nn.Module: The model fitted with the best weights found by the genetic algorithm.
        """
        population = []
        # Initialize population randomly.
        for idx in range(self.population_count):
            random_weights = torch.rand(
                self.count_of_initial_model_weights, device=self.device
            )
            population.append(random_weights)
        # Train weights with genetic algorithm.
        for idx in tqdm(range(num_iterations)):
            best_individuals_in_population = self.natural_selection(
                population, x_train, y_train
            )
            children = self.cross_over(best_individuals_in_population)
            mutated_children = self.mutation(children)
            new_population = best_individuals_in_population + mutated_children
        # Select the best individual evolution has created.
        best_state_dict_one_dimensional = self.natural_selection(
            new_population, x_train, y_train
        )[0]
        best_state_dict = self.reconstruct_statedict(best_state_dict_one_dimensional)
        # Load the trained weights to model.
        self.model.load_state_dict(best_state_dict)
        return self.model
