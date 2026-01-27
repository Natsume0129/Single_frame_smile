import torch

p1 = r"E:\Single_frame_smile\data\models\vggface.pth"
p2 = r"E:\Single_frame_smile\data\models\vggface_conv.pth"

sd1 = torch.load(p1, map_location="cpu")
sd2 = torch.load(p2, map_location="cpu")

k1 = set(sd1.keys())
k2 = set(sd2.keys())

print("Only in vggface.pth:")
for k in sorted(list(k1 - k2)):
    print(" ", k)

print("\nOnly in vggface_conv.pth:")
for k in sorted(list(k2 - k1)):
    print(" ", k)
