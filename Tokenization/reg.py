import regex as re
import tiktoken

gpt2pat = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# print(re.findall(gpt2pat, "Hello've world123 how are you"))



# Fixed the variable name from 'nc' to 'enc'
enc = tiktoken.get_encoding("gpt2") 
print(enc.encode("Hello World ")) 

# This block works correctly
enc2 = tiktoken.get_encoding("cl100k_base") 
print(enc2.encode("Hello World "))
