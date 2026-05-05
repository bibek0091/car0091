import difflib

def gen_diff(f1, f2):
    with open(f1, encoding='utf-8') as a, open(f2, encoding='utf-8') as b:
        diff = list(difflib.unified_diff(a.readlines(), b.readlines(), n=0))
        if not diff:
            return ""
        s = f"### {f1.split('\\\\')[-1]}\n```diff\n"
        for line in diff:
            s += line
        s += "```\n"
        return s

s = gen_diff(r"C:\Users\p23mi\Documents\car0091\perception\lane_detector.py", r"C:\Users\p23mi\Documents\BFMC_2026 - Copy\src\perception\lane_detector.py")
s += gen_diff(r"C:\Users\p23mi\Documents\car0091\perception\lane_tracker.py", r"C:\Users\p23mi\Documents\BFMC_2026 - Copy\src\perception\lane_tracker.py")
s += gen_diff(r"C:\Users\p23mi\Documents\car0091\control\controller.py", r"C:\Users\p23mi\Documents\BFMC_2026 - Copy\src\control\controller.py")

with open(r"C:\Users\p23mi\Documents\car0091\diff_out.txt", "w", encoding='utf-8') as f:
    f.write(s)
