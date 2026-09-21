"""The 20 tasks of the hardcode suite (see hardcode_eval.py).

Each task is a dict: id, prompt, reference (a validated solution) and tests (hidden
asserts run after the candidate; `__SRC__` holds the candidate source, so a test can
forbid e.g. eval or itertools). References and tests are validated by
`hardcode_eval.py --self-test` — change them together or not at all.
"""
from __future__ import annotations

SUFFIX = '\nReturn only one ```python code block, no explanation.'

TASKS: list[dict[str, str]] = []


def _add(task_id: str, prompt: str, reference: str, tests: str) -> None:
    TASKS.append({"id": task_id, "prompt": prompt + SUFFIX, "reference": reference, "tests": tests})


_add(
    'h01',
    "Write `lru_cache_sim(capacity: int, ops: list[tuple]) -> list` that simulates an LRU cache. ops are ('put', key, value) or ('get', key). Return the list of results of every 'get' (the value, or -1 if absent). A 'get' refreshes recency; 'put' of an existing key updates and refreshes; when over capacity evict the least recently used.",
    """from collections import OrderedDict
def lru_cache_sim(capacity, ops):
    d=OrderedDict(); out=[]
    for op in ops:
        if op[0]=='get':
            k=op[1]
            if k in d: d.move_to_end(k); out.append(d[k])
            else: out.append(-1)
        else:
            _,k,v=op
            if k in d: d.move_to_end(k)
            d[k]=v
            if len(d)>capacity: d.popitem(last=False)
    return out""",
    """assert lru_cache_sim(2,[('put',1,1),('put',2,2),('get',1),('put',3,3),('get',2),('put',4,4),('get',1),('get',3),('get',4)])==[1,-1,-1,3,4]
assert lru_cache_sim(1,[('put',1,5),('put',1,6),('get',1),('put',2,7),('get',1)])==[6,-1]
assert lru_cache_sim(2,[('get',9)])==[-1]""",
)

_add(
    'h02',
    'Write `eval_expr(s: str) -> float` that evaluates an arithmetic expression with + - * / , parentheses, unary minus, integers and decimals, and arbitrary spaces, using normal precedence. Do not use eval/exec/ast.',
    """def eval_expr(s):
    s=s.replace(' ',''); i=0
    def expr():
        nonlocal i; v=term()
        while i<len(s) and s[i] in '+-':
            o=s[i]; i+=1; r=term(); v=v+r if o=='+' else v-r
        return v
    def term():
        nonlocal i; v=fac()
        while i<len(s) and s[i] in '*/':
            o=s[i]; i+=1; r=fac(); v=v*r if o=='*' else v/r
        return v
    def fac():
        nonlocal i
        if s[i]=='-': i+=1; return -fac()
        if s[i]=='+': i+=1; return fac()
        if s[i]=='(':
            i+=1; v=expr(); i+=1; return v
        j=i
        while i<len(s) and (s[i].isdigit() or s[i]=='.'): i+=1
        return float(s[j:i])
    return expr()""",
    """assert abs(eval_expr('1 + 2 * 3')-7)<1e-9
assert abs(eval_expr('(1+2)*3')-9)<1e-9
assert abs(eval_expr('-(2+3)*2 + 10/4')-(-7.5))<1e-9
assert abs(eval_expr('2*-3')-(-6))<1e-9
assert abs(eval_expr('1.5*(2 - 0.5)/3')-0.75)<1e-9
assert 'eval(' not in __SRC__ and 'exec(' not in __SRC__""",
)

_add(
    'h03',
    'Write `topo_sort(n: int, edges: list[tuple[int,int]]) -> list[int] | None`. Nodes are 0..n-1, an edge (a,b) means a must come before b. Return the lexicographically smallest valid topological order, or None if there is a cycle.',
    """import heapq
def topo_sort(n, edges):
    g=[[] for _ in range(n)]; ind=[0]*n
    for a,b in edges: g[a].append(b); ind[b]+=1
    h=[i for i in range(n) if ind[i]==0]; heapq.heapify(h); out=[]
    while h:
        u=heapq.heappop(h); out.append(u)
        for v in g[u]:
            ind[v]-=1
            if ind[v]==0: heapq.heappush(h,v)
    return out if len(out)==n else None""",
    """assert topo_sort(4,[(1,0),(2,0),(3,1)])==[2,3,1,0]
assert topo_sort(3,[(0,1),(1,2),(2,0)]) is None
assert topo_sort(3,[])==[0,1,2]
assert topo_sort(5,[(4,0),(4,1),(0,2),(1,2),(2,3)])==[4,0,1,2,3]""",
)

_add(
    'h04',
    'Write `min_window(s: str, t: str) -> str` returning the smallest substring of s that contains every character of t with multiplicity (empty string if none). If several have the same length return the leftmost.',
    """from collections import Counter
def min_window(s,t):
    if not t or not s: return ''
    need=Counter(t); miss=len(t); l=0; best=(0,None,None)
    for r,c in enumerate(s):
        if need[c]>0: miss-=1
        need[c]-=1
        while miss==0:
            if best[1] is None or r-l+1<best[0]: best=(r-l+1,l,r)
            need[s[l]]+=1
            if need[s[l]]>0: miss+=1
            l+=1
    return '' if best[1] is None else s[best[1]:best[2]+1]""",
    """assert min_window('ADOBECODEBANC','ABC')=='BANC'
assert min_window('a','aa')==''
assert min_window('aa','aa')=='aa'
assert min_window('abcabdebac','cda')=='cabd'
assert min_window('xyz','')==''""",
)

_add(
    'h05',
    "Write `dijkstra(n: int, edges: list[tuple[int,int,int]], src: int) -> list[float]` for a directed graph with non-negative weights (edges are (u, v, w)). Return the list of shortest distances from src to every node 0..n-1, using float('inf') for unreachable nodes.",
    """import heapq
def dijkstra(n,edges,src):
    g=[[] for _ in range(n)]
    for u,v,w in edges: g[u].append((v,w))
    d=[float('inf')]*n; d[src]=0; h=[(0,src)]
    while h:
        du,u=heapq.heappop(h)
        if du>d[u]: continue
        for v,w in g[u]:
            if du+w<d[v]: d[v]=du+w; heapq.heappush(h,(d[v],v))
    return d""",
    """assert dijkstra(5,[(0,1,4),(0,2,1),(2,1,2),(1,3,1),(2,3,5)],0)==[0,3,1,4,float('inf')]
assert dijkstra(1,[],0)==[0]
assert dijkstra(3,[(0,1,0),(1,2,0)],0)==[0,0,0]""",
)

_add(
    'h06',
    'Write `longest_increasing_subsequence(a: list[int]) -> list[int]` returning one longest strictly increasing subsequence in O(n log n). If several exist, return the one that ends with the smallest possible last value; any valid one of maximum length with that property is accepted.',
    """import bisect
def longest_increasing_subsequence(a):
    tails=[]; idx=[]; prev=[-1]*len(a)
    for i,x in enumerate(a):
        j=bisect.bisect_left(tails,x)
        if j==len(tails): tails.append(x); idx.append(i)
        else: tails[j]=x; idx[j]=i
        prev[i]=idx[j-1] if j>0 else -1
    out=[]; k=idx[-1] if idx else -1
    while k!=-1: out.append(a[k]); k=prev[k]
    return out[::-1]""",
    """def _ok(a,L):
    r=longest_increasing_subsequence(a); assert len(r)==L, (a,r); assert all(x<y for x,y in zip(r,r[1:]));
    it=iter(a); assert all(any(x==y for y in it) for x in r)
_ok([10,9,2,5,3,7,101,18],4)
_ok([],0)
_ok([5,5,5],1)
_ok([1,3,6,7,9,4,10,5,6],6)
_ok(list(range(2000,0,-1))+list(range(3000)),3000)""",
)

_add(
    'h07',
    'Write `merge_k_sorted(lists: list[list[int]]) -> list[int]` that merges k sorted lists into one sorted list in O(N log k) using a heap (do not just concatenate and sort).',
    """import heapq
def merge_k_sorted(lists):
    h=[(l[0],i,0) for i,l in enumerate(lists) if l]; heapq.heapify(h); out=[]
    while h:
        v,i,j=heapq.heappop(h); out.append(v)
        if j+1<len(lists[i]): heapq.heappush(h,(lists[i][j+1],i,j+1))
    return out""",
    """assert merge_k_sorted([[1,4,5],[1,3,4],[2,6]])==[1,1,2,3,4,4,5,6]
assert merge_k_sorted([])==[]
assert merge_k_sorted([[],[1],[]])==[1]
assert 'heapq' in __SRC__ or 'heap' in __SRC__""",
)

_add(
    'h08',
    'Write `parse_csv_line(line: str) -> list[str]` that splits one CSV line per RFC 4180: fields separated by commas, fields may be wrapped in double quotes, quoted fields may contain commas, and a doubled double-quote inside a quoted field is a literal quote. Do not use the csv module.',
    """def parse_csv_line(line):
    out=[]; cur=[]; q=False; i=0
    while i<len(line):
        c=line[i]
        if q:
            if c=='"':
                if i+1<len(line) and line[i+1]=='"': cur.append('"'); i+=1
                else: q=False
            else: cur.append(c)
        else:
            if c==',': out.append(''.join(cur)); cur=[]
            elif c=='"': q=True
            else: cur.append(c)
        i+=1
    out.append(''.join(cur)); return out""",
    (
    "assert parse_csv_line('a,b,c')==['a','b','c']\n"
    'assert parse_csv_line(\'a,"b,c",d\')==[\'a\',\'b,c\',\'d\']\n'
    'assert parse_csv_line(\'"he said ""hi""",x\')==[\'he said "hi"\',\'x\']\n'
    "assert parse_csv_line(',,')==['','','']\n"
    "assert parse_csv_line('')==['']\n"
    "assert 'import csv' not in __SRC__"
),
)

_add(
    'h09',
    "Write `count_islands(grid: list[list[int]]) -> int` counting groups of 1s connected 4-directionally. It must not hit Python's recursion limit on a 300x300 grid of all 1s.",
    """def count_islands(grid):
    if not grid: return 0
    R,C=len(grid),len(grid[0]); seen=[[False]*C for _ in range(R)]; n=0
    for r in range(R):
        for c in range(C):
            if grid[r][c]==1 and not seen[r][c]:
                n+=1; st=[(r,c)]; seen[r][c]=True
                while st:
                    y,x=st.pop()
                    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        a,b=y+dy,x+dx
                        if 0<=a<R and 0<=b<C and grid[a][b]==1 and not seen[a][b]: seen[a][b]=True; st.append((a,b))
    return n""",
    """assert count_islands([[1,1,0],[0,1,0],[0,0,1]])==2
assert count_islands([])==0
assert count_islands([[0]])==0
assert count_islands([[1]*300 for _ in range(300)])==1
assert count_islands([[1,0,1],[0,1,0],[1,0,1]])==5""",
)

_add(
    'h10',
    'Write `RateLimiter` class: `__init__(self, max_calls: int, window: float)` and `allow(self, now: float) -> bool`. It is a sliding-window limiter: allow returns True and records the call if fewer than max_calls calls were recorded in the half-open interval (now - window, now], else False. `now` values are non-decreasing.',
    """from collections import deque
class RateLimiter:
    def __init__(self,max_calls,window): self.m=max_calls; self.w=window; self.q=deque()
    def allow(self,now):
        while self.q and self.q[0]<=now-self.w: self.q.popleft()
        if len(self.q)<self.m: self.q.append(now); return True
        return False""",
    """r=RateLimiter(2,10.0)
assert [r.allow(t) for t in (0,1,2,10,10.5,11,12)]==[True,True,False,True,False,True,False]
r=RateLimiter(1,1.0)
assert [r.allow(t) for t in (0,0.5,1.0,1.0)]==[True,False,True,False]""",
)

_add(
    'h11',
    'Write `knapsack(weights: list[int], values: list[int], capacity: int) -> tuple[int, list[int]]` for the 0/1 knapsack. Return (best total value, sorted list of chosen item indices). If several optimal sets exist, any is accepted.',
    """def knapsack(weights,values,capacity):
    n=len(weights); dp=[[0]*(capacity+1) for _ in range(n+1)]
    for i in range(1,n+1):
        w,v=weights[i-1],values[i-1]
        for c in range(capacity+1):
            dp[i][c]=dp[i-1][c]
            if c>=w and dp[i-1][c-w]+v>dp[i][c]: dp[i][c]=dp[i-1][c-w]+v
    c=capacity; ch=[]
    for i in range(n,0,-1):
        if dp[i][c]!=dp[i-1][c]: ch.append(i-1); c-=weights[i-1]
    return dp[n][capacity],sorted(ch)""",
    """def _ok(w,v,c,best):
    val,ch=knapsack(w,v,c); assert val==best,(val,best); assert ch==sorted(set(ch)); assert sum(w[i] for i in ch)<=c; assert sum(v[i] for i in ch)==best
_ok([1,3,4,5],[1,4,5,7],7,9)
_ok([],[],5,0)
_ok([5],[10],4,0)
_ok([2,2,2,2],[3,3,3,3],5,6)
_ok([10,20,30],[60,100,120],50,220)""",
)

_add(
    'h12',
    "Write `format_table(rows: list[list[str]]) -> str` that renders rows as a text table: each column is as wide as its longest cell, cells are left-aligned and padded with spaces, columns are joined with ' | ', there is no trailing whitespace on any line, the first row is a header followed by a separator line made of '-' characters for each column joined with '-+-', and lines are joined with '\\n'. Rows may have different lengths; missing cells are empty.",
    r"""def format_table(rows):
    if not rows: return ''
    n=max(len(r) for r in rows); rows=[list(r)+['']*(n-len(r)) for r in rows]
    w=[max(len(r[i]) for r in rows) for i in range(n)]
    def line(r): return ' | '.join(c.ljust(w[i]) for i,c in enumerate(r)).rstrip()
    out=[line(rows[0]),'-+-'.join('-'*x for x in w)]+[line(r) for r in rows[1:]]
    return '\n'.join(out)""",
    r"""assert format_table([['name','qty'],['bolt','40'],['washer','5']])=='name   | qty\n-------+----\nbolt   | 40\nwasher | 5'
assert format_table([['a','b','c'],['1']])=='a | b | c\n--+---+--\n1 |   |'
assert format_table([])==''""",
)

_add(
    'h13',
    'Write `next_permutation(a: list[int]) -> list[int] | None` returning the next lexicographic permutation of a as a new list (input may contain duplicates), or None if a is already the last permutation. Do not use itertools.',
    """def next_permutation(a):
    a=list(a); i=len(a)-2
    while i>=0 and a[i]>=a[i+1]: i-=1
    if i<0: return None
    j=len(a)-1
    while a[j]<=a[i]: j-=1
    a[i],a[j]=a[j],a[i]; a[i+1:]=reversed(a[i+1:]); return a""",
    """assert next_permutation([1,2,3])==[1,3,2]
assert next_permutation([3,2,1]) is None
assert next_permutation([1,1,5])==[1,5,1]
assert next_permutation([1,3,2])==[2,1,3]
assert next_permutation([2,2,2]) is None
assert next_permutation([])is None
x=[1,2,3]; next_permutation(x); assert x==[1,2,3]
assert 'itertools' not in __SRC__""",
)

_add(
    'h14',
    'Write `Trie` class with `insert(word: str)`, `search(word: str) -> bool` (exact word), `starts_with(prefix: str) -> bool`, and `delete(word: str) -> bool` (returns False if the word was not present; after deletion prefixes that no longer lead to any word must make starts_with return False).',
    """class Trie:
    def __init__(self): self.c={}; self.end=False
    def insert(self,w):
        n=self
        for ch in w: n=n.c.setdefault(ch,Trie())
        n.end=True
    def _find(self,w):
        n=self
        for ch in w:
            if ch not in n.c: return None
            n=n.c[ch]
        return n
    def search(self,w):
        n=self._find(w); return bool(n and n.end)
    def starts_with(self,p):
        n=self._find(p); return n is not None and (n.end or bool(n.c)) if p else (self.end or bool(self.c))
    def delete(self,w):
        path=[self]; n=self
        for ch in w:
            if ch not in n.c: return False
            n=n.c[ch]; path.append(n)
        if not n.end: return False
        n.end=False
        for i in range(len(w)-1,-1,-1):
            node=path[i+1]
            if node.end or node.c: break
            del path[i].c[w[i]]
        return True""",
    """t=Trie(); t.insert('apple'); t.insert('app')
assert t.search('app') and t.search('apple') and not t.search('appl')
assert t.starts_with('appl')
assert t.delete('apple') and not t.search('apple') and t.search('app')
assert not t.starts_with('appl') and t.starts_with('ap')
assert not t.delete('apple') and not t.delete('zzz')
assert t.delete('app') and not t.starts_with('a')""",
)

_add(
    'h15',
    "Write `roman_to_int(s: str) -> int` and `int_to_roman(n: int) -> str` for 1..3999. roman_to_int must raise ValueError for strings that are not the canonical roman numeral of some number in range (for example 'IIII', 'VX', 'IC', '').",
    """def int_to_roman(n):
    if not 1<=n<=3999: raise ValueError(n)
    out=[]
    for v,s in ((1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')):
        while n>=v: out.append(s); n-=v
    return ''.join(out)
def roman_to_int(s):
    m={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
    if not s or any(c not in m for c in s): raise ValueError(s)
    t=0
    for i,c in enumerate(s):
        if i+1<len(s) and m[c]<m[s[i+1]]: t-=m[c]
        else: t+=m[c]
    if not 1<=t<=3999 or int_to_roman(t)!=s: raise ValueError(s)
    return t""",
    """assert int_to_roman(1994)=='MCMXCIV' and roman_to_int('MCMXCIV')==1994
assert all(roman_to_int(int_to_roman(i))==i for i in range(1,4000))
for bad in ('IIII','VX','IC','','MMMM','abc','IL'):
    try:
        roman_to_int(bad); raise AssertionError(bad)
    except ValueError: pass""",
)

_add(
    'h16',
    'Write `schedule_meetings(meetings: list[tuple[int,int]]) -> int` returning the minimum number of rooms needed so that no two meetings in the same room overlap. Meetings are half-open intervals [start, end): a meeting ending at t does not conflict with one starting at t.',
    """import heapq
def schedule_meetings(meetings):
    h=[]; best=0
    for s,e in sorted(meetings):
        while h and h[0]<=s: heapq.heappop(h)
        heapq.heappush(h,e); best=max(best,len(h))
    return best""",
    """assert schedule_meetings([(0,30),(5,10),(15,20)])==2
assert schedule_meetings([(7,10),(2,4)])==1
assert schedule_meetings([])==0
assert schedule_meetings([(1,5),(5,9),(9,12)])==1
assert schedule_meetings([(1,10),(2,9),(3,8),(4,7)])==4""",
)

_add(
    'h17',
    "Write `diff_lines(a: list[str], b: list[str]) -> list[str]` producing a minimal line diff based on the longest common subsequence: output lines prefixed with '  ' for unchanged, '- ' for lines only in a, '+ ' for lines only in b, in order. When a deletion and an insertion are adjacent, deletions come first.",
    """def diff_lines(a,b):
    n,m=len(a),len(b); L=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n-1,-1,-1):
        for j in range(m-1,-1,-1):
            L[i][j]=L[i+1][j+1]+1 if a[i]==b[j] else max(L[i+1][j],L[i][j+1])
    i=j=0; out=[]
    while i<n and j<m:
        if a[i]==b[j]: out.append('  '+a[i]); i+=1; j+=1
        elif L[i+1][j]>=L[i][j+1]: out.append('- '+a[i]); i+=1
        else: out.append('+ '+b[j]); j+=1
    out+=['- '+x for x in a[i:]]; out+=['+ '+x for x in b[j:]]; return out""",
    """def _ok(a,b,nsame):
    d=diff_lines(a,b); assert [x[2:] for x in d if x[:2] in ('  ','- ')]==a; assert [x[2:] for x in d if x[:2] in ('  ','+ ')]==b; assert sum(1 for x in d if x[:2]=='  ')==nsame,(d,nsame); assert all(x[:2] in ('  ','- ','+ ') for x in d)
_ok(['a','b','c'],['a','c','d'],2)
_ok([],['x'],0)
_ok(['x'],[],0)
_ok(['a','b','c','d','e'],['b','d','e','f'],3)
assert diff_lines(['a'],['b'])==['- a','+ b']""",
)

_add(
    'h18',
    'Write `solve_sudoku(board: list[list[int]]) -> list[list[int]] | None` that solves a 9x9 sudoku (0 = empty) and returns the solved board as a new list of lists, or None if there is no solution. It must solve a hard puzzle in a few seconds.',
    """def solve_sudoku(board):
    b=[r[:] for r in board]
    rows=[set() for _ in range(9)]; cols=[set() for _ in range(9)]; box=[set() for _ in range(9)]
    for r in range(9):
        for c in range(9):
            v=b[r][c]
            if v:
                if v in rows[r] or v in cols[c] or v in box[r//3*3+c//3]: return None
                rows[r].add(v); cols[c].add(v); box[r//3*3+c//3].add(v)
    def go():
        best=None; bc=None
        for r in range(9):
            for c in range(9):
                if b[r][c]==0:
                    cand=[v for v in range(1,10) if v not in rows[r] and v not in cols[c] and v not in box[r//3*3+c//3]]
                    if not cand: return False
                    if best is None or len(cand)<len(bc): best=(r,c); bc=cand
        if best is None: return True
        r,c=best
        for v in bc:
            b[r][c]=v; rows[r].add(v); cols[c].add(v); box[r//3*3+c//3].add(v)
            if go(): return True
            b[r][c]=0; rows[r].discard(v); cols[c].discard(v); box[r//3*3+c//3].discard(v)
        return False
    return b if go() else None""",
    """P=[[0,0,0,0,0,0,0,1,2],[0,0,0,0,3,5,0,0,0],[0,0,0,6,0,0,0,7,0],[7,0,0,0,0,0,3,0,0],[0,0,0,4,0,0,8,0,0],[1,0,0,0,0,0,0,0,0],[0,0,0,1,2,0,0,0,0],[0,8,0,0,0,0,0,4,0],[0,5,0,0,0,0,6,0,0]]
import copy; Q=copy.deepcopy(P); S=solve_sudoku(P)
assert P==Q
assert S is not None and all(sorted(r)==list(range(1,10)) for r in S) and all(sorted(S[r][c] for r in range(9))==list(range(1,10)) for c in range(9))
assert all(sorted(S[r][c] for r in range(br,br+3) for c in range(bc,bc+3))==list(range(1,10)) for br in (0,3,6) for bc in (0,3,6))
assert all(S[r][c]==P[r][c] for r in range(9) for c in range(9) if P[r][c])
B=[[5,5,0,0,0,0,0,0,0]]+[[0]*9 for _ in range(8)]
assert solve_sudoku(B) is None""",
)

_add(
    'h19',
    "Write `json_flatten(obj) -> dict` that flattens nested dicts and lists into a single-level dict with dotted keys, using `[i]` for list indices (for example {'a': {'b': [1, {'c': 2}]}} -> {'a.b[0]': 1, 'a.b[1].c': 2}). Empty dicts and empty lists are kept as values. Then write `json_unflatten(flat: dict)` that reverses it exactly.",
    r"""import re
def json_flatten(obj):
    out={}
    def go(o,p):
        if isinstance(o,dict) and o:
            for k,v in o.items(): go(v,f'{p}.{k}' if p else k)
        elif isinstance(o,list) and o:
            for i,v in enumerate(o): go(v,f'{p}[{i}]')
        else: out[p]=o
    go(obj,''); return out
def json_unflatten(flat):
    if list(flat.keys())==['']: return flat['']
    root=None
    def toks(k): return [int(a) if a else b for a,b in re.findall(r'\[(\d+)\]|([^.\[\]]+)',k)]
    for k,v in flat.items():
        t=toks(k)
        if root is None: root=[] if isinstance(t[0],int) else {}
        cur=root
        for i,x in enumerate(t):
            last=i==len(t)-1; nxt=None if last else ([] if isinstance(t[i+1],int) else {})
            if isinstance(x,int):
                while len(cur)<=x: cur.append(None)
                if last: cur[x]=v
                else:
                    if cur[x] is None: cur[x]=nxt
                    cur=cur[x]
            else:
                if last: cur[x]=v
                else:
                    if x not in cur: cur[x]=nxt
                    cur=cur[x]
    return root if root is not None else {}""",
    """A={'a':{'b':[1,{'c':2}],'d':{}},'e':[],'f':[[1,2],[3]],'g':None}
F=json_flatten(A)
assert F=={'a.b[0]':1,'a.b[1].c':2,'a.d':{},'e':[],'f[0][0]':1,'f[0][1]':2,'f[1][0]':3,'g':None},F
assert json_unflatten(F)==A
assert json_unflatten(json_flatten({'x':[{'y':[0,{'z':'q'}]}]}))=={'x':[{'y':[0,{'z':'q'}]}]}""",
)

_add(
    'h20',
    "Write `BankLedger` class. `transfer(src: str, dst: str, amount: int)` moves money atomically: it raises ValueError (and changes nothing) if amount <= 0, if src == dst, or if src has insufficient funds. `deposit(acct, amount)` adds funds (ValueError if amount <= 0). `balance(acct) -> int` returns 0 for unknown accounts. `history(acct) -> list[tuple[str,int]]` returns that account's entries in order as ('deposit', n), ('out', n) or ('in', n). `rollback(n: int)` undoes the last n successful operations (deposits or transfers) across the ledger, including their history entries; ValueError if n exceeds the number of operations.",
    """class BankLedger:
    def __init__(self): self.b={}; self.h={}; self.ops=[]
    def balance(self,a): return self.b.get(a,0)
    def history(self,a): return list(self.h.get(a,[]))
    def deposit(self,a,n):
        if n<=0: raise ValueError
        self.b[a]=self.balance(a)+n; self.h.setdefault(a,[]).append(('deposit',n)); self.ops.append(('d',a,n))
    def transfer(self,s,d,n):
        if n<=0 or s==d or self.balance(s)<n: raise ValueError
        self.b[s]=self.balance(s)-n; self.b[d]=self.balance(d)+n
        self.h.setdefault(s,[]).append(('out',n)); self.h.setdefault(d,[]).append(('in',n)); self.ops.append(('t',s,d,n))
    def rollback(self,n):
        if n>len(self.ops) or n<0: raise ValueError
        for _ in range(n):
            op=self.ops.pop()
            if op[0]=='d': _,a,x=op; self.b[a]-=x; self.h[a].pop()
            else: _,s,d,x=op; self.b[s]+=x; self.b[d]-=x; self.h[s].pop(); self.h[d].pop()""",
    """L=BankLedger(); L.deposit('a',100); L.transfer('a','b',30)
assert (L.balance('a'),L.balance('b'),L.balance('zz'))==(70,30,0)
for bad in (lambda:L.transfer('a','b',0),lambda:L.transfer('a','a',5),lambda:L.transfer('b','a',31),lambda:L.deposit('a',-1),lambda:L.rollback(3)):
    try:
        bad(); raise AssertionError('no error')
    except ValueError: pass
assert (L.balance('a'),L.balance('b'))==(70,30)
assert L.history('a')==[('deposit',100),('out',30)] and L.history('b')==[('in',30)]
L.transfer('b','c',10); L.rollback(2)
assert (L.balance('a'),L.balance('b'),L.balance('c'))==(100,0,0) and L.history('b')==[] and L.history('a')==[('deposit',100)]""",
)
