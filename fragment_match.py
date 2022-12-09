import difflib
from typing import Tuple, List


def best_match(s1: str, s2: str) -> Tuple[
    float, int, List[int], Tuple[str, str], Tuple[str, str]]:
    '''
    return the best match of s1 in s2
    Args:
        s1:
        s2:

    Returns:
        ratio: value within 0-1
        start: the start of s2 mean the 0 of s1
        match_ids: id maps from s1 to s2  [-1 means not match]
        match: (s1, matched string in s2)
        match_core: (s1_core, match s2 core) core means strip("-")
    '''
    ls = 0
    for s in s1:
        if s != "-":
            break
        else:
            ls += 1
    rs = 0
    print(s1[::-1])
    for s in s1[::-1]:
        if s != "-":
            break
        else:
            rs += 1
    s1_new = s1.strip("-")
    ls1 = len(s1_new)
    s2_new = "#" * ls1 + s2 + "#" * ls1
    ls2 = len(s2_new)
    n_b = s1_new.count("-")
    ratios = [[] for _ in range(n_b + 1)]
    for n_bi in range(n_b + 1):
        for lb in range(ls2 - ls1 + 1):
            ratios[n_bi].append(difflib.SequenceMatcher(None, s1_new, s2_new[lb:lb + ls1 - n_bi]).quick_ratio())
    m_ij = [max(i) for i in ratios]
    m_i = max(m_ij)
    id_i = m_ij.index(m_i)
    id_j = ratios[id_i].index(m_i)
    select_s2_p = s2_new[id_j:id_j + ls1 - id_i]

    s1_ids = list(range(ls1))
    s2_ids = [-1] * (ls1 - id_i)
    # TODO: Because the transform only delete "-" and insert "-", so cannot use get_opcodes() methods.
    matchs = difflib.SequenceMatcher(None, s1_new, select_s2_p).get_matching_blocks()
    for m in matchs:
        if m.size > 2:
            s2_ids[m.b:m.b + m.size] = s1_ids[m.a:m.a + m.size]
    # TODO: set initial
    for id, i in enumerate(s2_ids):
        if i == -1:
            s2_ids[id] = id
        else:
            break
    # shift
    for id, i in enumerate(s2_ids):
        s2_ids[id] = i + ls
    s2_ids = ls * [-1] + s2_ids + rs * [-1]
    start = id_j - ls1 - ls
    match_ids = s2_ids
    match = (s1, s2[start:start + len(s2_ids)])
    match_core = (s1_new, select_s2_p)
    ratio = m_i
    return ratio, start, match_ids, match, match_core


if __name__ == '__main__':
    tm6 = "----------THRMRTETKAAKTLCIIMGCFCL-CWAPFFVTNIVDPFI----"
    fasta = "KVVLLTFLSTVILMAILGNLLVMVAVCWDRQLRKIKTNYFIVSLAFADLLVSVLVMPFGAIELVQDIWIYGEVFCLVRTSLDVLLTTASIFHLCCISLDRYYAICCQPLVYRNKMTPLRIALMLGGCWVIPTFISFLPIMQGWNNIGIIDLIEKRKFNQNSNSTYCVFMVNKPYAITCSVVAFYIPFLLMVLAYYRIYVTAKEHAHQIQMLQRAGASSETKAAKTLCIIMGCFCLCWAPFFVTNIVDPFIDYTVPGQVWTAFLWLGYINSGLNPFLYAFLNKSFRRAFLIILCC"
    match = best_match(tm6, fasta)
    # print(match[0])
    # print(match[1])
    # print(match[2])
    # print(match[3])
    # print(match[4])
