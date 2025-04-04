from torch.utils.data import Subset

# used for teacher allocating
def teacher_allocate(teacher_lst, k):
    n = len(teacher_lst)
    part_size = n // k
    remainder = n % k  # Handle cases where n is not perfectly divisible

    parts = []
    start = 0
    for i in range(k):
        extra = 1 if i < remainder else 0  # Distribute remaining elements
        end = start + part_size + extra
        parts.append(teacher_lst[start:end])
        start = end
    return parts


# divide chunk into slices
def chunk_slice(chunk_set, r):
    sliced_chunk = []
    for chunk in chunk_set:
        slice_set = []
        chunk_size = len(chunk)
        slice_size = chunk_size//r
        sizes = [slice_size] * r
        sizes[-1] += chunk_size % r
        start = 0
        for size in sizes:
            slice = Subset(chunk, range(start, start+size))
            start = start + size
            slice_set.append(slice)
        sliced_chunk.append(slice_set)
    
    return sliced_chunk
