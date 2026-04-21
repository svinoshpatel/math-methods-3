import json
import numpy as np

def save_tasks_to_file(filename="tasks.json"):
    tasks = []
    for i in range(20):
        # Варіюємо розміри
        if i < 5: r, c = np.random.randint(3, 6), np.random.randint(3, 6)
        elif i < 15: r, c = np.random.randint(10, 30), np.random.randint(10, 30)
        else: r, c = np.random.randint(80, 105), np.random.randint(80, 105)
        
        costs = np.random.randint(1, 50, size=(r, c)).tolist()
        supply = np.random.randint(20, 100, size=r).tolist()
        demand = np.random.randint(20, 100, size=c).tolist()
        
        # Балансування
        s_sum, d_sum = sum(supply), sum(demand)
        diff = s_sum - d_sum
        demand[-1] = max(1, demand[-1] + diff)
        # Фінальна корекція суми
        supply[-1] += (sum(demand) - sum(supply))
        
        tasks.append({
            "id": i + 1,
            "costs": costs,
            "supply": supply,
            "demand": demand
        })
    
    with open(filename, "w") as f:
        json.dump(tasks, f)
    print(f"Файл {filename} створено!")

if __name__ == "__main__":
    save_tasks_to_file()
