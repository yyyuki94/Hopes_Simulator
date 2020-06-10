from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np

# GA選択の抽象クラス
class Selection(metaclass=ABCMeta):
    @abstractmethod
    def select(self, fitness: np.ndarray) -> int:
        pass
    
    
# GAのルーレット選択
class RouletteSelection(Selection):
    def select(self, fitness):
        prob = fitness / np.sum(fitness)
        
        selected = np.random.choice(range(len(prob)), size=1, p=prob)
        return selected
            
        
# GA交叉の抽象クラス
class Crossover(metaclass=ABCMeta):
    @abstractmethod
    def crossover(self, idv1: np.ndarray, idv2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(idv1) != len(idv2):
            raise TypeError(f"Length of idv1 and idv2 should be equal")
        
    
# GAの一様交叉
class UniformCossover(Crossover):
    def crossover(self, idv1, idv2):
        mask = np.random.randint(0, 2, len(idv1), dtype=bool)
        idvA, idvB = idv1.copy(), idv2.copy()
        idvA[mask] = idv2[mask].copy()
        idvB[mask] = idv1[mask].copy()
        
        return idvA, idvB
        

# GAのk点交叉
class KPointCrossover(Crossover):
    def __init__(self, k=2):
        self.__k = k
    
    def crossover(self, idv1, idv2):
        points = np.random.choice(range(len(idv1)), size=self.__k, replace=False)
        points = np.sort(points)
        mask = np.zeros(len(idv1), dtype=bool)
        
        bl = True
        for i in range(len(points) - 1):
            mask[points[i]:points[i+1]] = bl
            bl = not(bl)
            
        idvA, idvB = idv1.copy(), idv2.copy()
        idvA[mask] = idv2[mask].copy()
        idvB[mask] = idv1[mask].copy()
        
        return idvA, idvB

            
# GA突然変異の抽象クラス
class Mutation(metaclass=ABCMeta):
    @abstractmethod
    def mutate(self, idv: np.ndarray) -> np.ndarray:
        pass

    
# GAの1ビット突然変異
class BitStringMutation(Mutation):
    def mutate(self, idv):
        p = np.random.randint(len(idv))
        mutated = idv.copy()
        mutated[p] = not(mutated[p])
        
        return mutated


# GAの個体郡クラス
class Population:
    def __init__(self, N, m, fitness_func, idvs=None):
        if not callable(fitness_func):
            raise TypeError(f"fitness_func must be callable.")
        
        if idvs is None:
            self.__individuals = np.random.randint(0, 2, (N, m), dtype=bool)
        else:
            if isinstance(idvs, np.ndarray) and idvs.shape == (N, m):
                self.__individuals = idvs.copy()
            else:
                raise TypeError(f"Type of idvs should be {type(np.ndarray)}. Or, shape of idvs should be {(N, m)}")
                
        self.__func = fitness_func
        self.__fitness_list = self.calc_fitness()

                
    def get_individuals(self):
        return self.__individuals
    
    def get_fitness(self):
        return self.__fitness_list
    
    def calc_fitness(self):
        return np.apply_along_axis(self.__func, 1, self.__individuals)


class GA:
    def __init__(self, selection: Selection, crossover: Crossover, mutation: Mutation,
                 m: int, fitness_func: callable, N=100, G=50, p_c=0.75, p_m=0.001, idvs=None):
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.fitness_func = fitness_func
        self.__N = N
        self.__G = G
        self.__m = m
        self.p_c = p_c
        self.p_m = p_m
        self.p_r = 1.0 - (p_c + p_m)
        self.__current_gen = Population(self.__N, self.__m, self.fitness_func, idvs=idvs)
        self.__next_gen = np.zeros((self.__N, self.__m), dtype=bool)
    
    def get_N(self):
        return self.__N
    
    def get_current_gen(self):
        return self.__current_gen
    
    def get_next_gen(self):
        return self.__next_gen
    
    def get_probs(self):
        return (self.p_c, self.p_m, self.p_r)

    def genetic_manupilate(self):
        idvs = self.__current_gen.get_individuals()
        fitness = self.__current_gen.get_fitness()
        select_func = self.selection.select
        null_ahed = np.where(self.__next_gen.any(axis=1) == False)[0][0]
        
        # mode; 0: 交叉, 1: 突然変異, 2: 再生
        mode = np.random.choice([0,1,2], p=self.get_probs())
        
        if mode == 0:
            # 交叉
            while True:
                idx1, idx2 = [select_func(fitness) for _ in range(2)]
                if idx1 != idx2:
                    break
            idvA, idvB = self.crossover.crossover(idvs[idx1, :][0], idvs[idx2, :][0])
            
            if null_ahed == self.__N - 1:
                self.__next_gen[null_ahed, :] = idvA.copy()
            else:
                self.__next_gen[null_ahed:null_ahed+2, :] = np.array([idvA, idvB]).copy()
            
        elif mode == 1:
            # 突然変異
            idv = idvs[select_func(fitness)][0]
            self.__next_gen[null_ahed, :] = self.mutation.mutate(idv).copy()
        else:
            # 再生
            self.__next_gen[null_ahed, :] = idvs[select_func(fitness)][0]
            
    def print_top5(self):
        idvs = self.__current_gen.get_individuals()
        fitness = self.__current_gen.get_fitness()
        sorted_idx = np.argsort(fitness)[::-1]
        
        default_np = np.get_printoptions()
        np.set_printoptions(threshold=0, edgeitems=6)
        
        for idx in sorted_idx[:5]:
            print(f"{idx}; Fit: {fitness[idx]:.05f}, Idv: {idvs[idx]}")
            
        np.set_printoptions(default_np)
    
    def eval_ga(self):
        for i in range(self.__G):
            print("="*25 + f" Generation {i+1:02} " + "="*25)
            self.print_top5()
            while not(self.__next_gen.any(axis=1).all()):
                self.genetic_manupilate()
            print("="*25 + f"===============" + "="*25)
            self.__current_gen = Population(self.__N, self.__m, self.fitness_func,
                                            idvs=self.__next_gen)
            self.__next_gen = np.zeros((self.__N, self.__m), dtype=bool)
            
        print("="*25 + f"    Result     " + "="*25)
        self.print_top5()
        
    def get_best_idvs(self):
        idvs = self.__current_gen.get_individuals()
        fitness = self.__current_gen.get_fitness()
        sorted_idx = np.argsort(fitness)[::-1]
        
        return idvs[sorted_idx[0], :]