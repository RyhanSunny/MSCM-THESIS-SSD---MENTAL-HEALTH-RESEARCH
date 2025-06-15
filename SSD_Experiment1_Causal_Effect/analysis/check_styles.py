import matplotlib.pyplot as plt
print("Available styles:")
for style in sorted(plt.style.available):
    print(f"  {style}")