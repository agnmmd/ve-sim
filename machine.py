import simpy
import copy
import numpy as np

class Product:
    def __init__(self, id):
        self.id = id
        self.processed = False


class Machine:
    def __init__(
        self, 
        env: simpy.Environment, 
        id, 
        process_time=(10, 15, 20)
    ):
        self.env = env
        self.id = id
        self.process_time = process_time
        self.idle = True

        self.pending_products = []
        self.processed_products = []

    def generate_process_time(self):
        return np.random.triangular(
            self.process_time[0], self.process_time[1], self.process_time[2]
        )

    def process_product(self, product):
        print(f"Processing product {product.id} at time {self.env.now}")
        self.idle = False
        yield self.env.timeout(delay=self.generate_process_time())
        product.processed = True
        self.idle = False
        return product

    def _set_production_plan(self, products):
        self.pending_products = copy.deepcopy(products)
        self.processed_products = []

    def process_all_products_randomly(self, products):
        self._set_production_plan(products)
        while len(self.pending_products) > 0:
            next_prod = np.random.choice(range(len(self.pending_products)))
            p = self.pending_products[next_prod]
            yield env.process(m.process_product(p))
            self.pending_products.remove(p)
            self.processed_products.append(p)

    def process_all_products_fifo(self, products):
        self._set_production_plan(products)
        while len(self.pending_products) > 0:
            p = self.pending_products[0]
            yield env.process(m.process_product(p))
            self.pending_products.remove(p)
            self.processed_products.append(p)

    def process_all_products_lifo(self, products):
        self._set_production_plan(products)
        while len(self.pending_products) > 0:
            p = self.pending_products[-1]
            yield env.process(m.process_product(p))
            self.pending_products.remove(p)
            self.processed_products.append(p)

if __name__ == "__main__":

    print("Processing RANDOM policy")
    env = simpy.Environment()
    m = Machine(env, id="machine")
    products = [Product(id=i) for i in range(5)]
    env.process(m.process_all_products_randomly(products))
    env.run()