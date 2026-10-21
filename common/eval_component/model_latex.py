import ast
import re

# converts a python model expression into latex (matplotlib mathtext subset), keeping the structure as written

greek = {"alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta", "theta", "iota", "kappa",
         "lambda", "mu", "nu", "xi", "rho", "sigma", "tau", "upsilon", "phi", "varphi", "chi", "psi", "omega",
         "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega"}

known_symbols = {"freq": "f", "pi": r"\pi", "c_thz": "c", "eps0_thz": r"\varepsilon_0",
                 "sig0": r"\sigma_0", "tau": r"\tau", "wp": r"\omega_p", "eps_inf": r"\varepsilon_\infty",
                 "eps_s": r"\varepsilon_s", "c1": "c_1", "eps": r"\varepsilon", "lam": r"\lambda"}

functions = {"sqrt": (r"\sqrt{", "}"), "abs": (r"\left|", r"\right|"), "conj": (r"\overline{", "}"),
             "exp": (r"\exp\left(", r"\right)"), "log": (r"\ln\left(", r"\right)"),
             "log10": (r"\log_{10}\left(", r"\right)"), "real": (r"\mathrm{Re}\left(", r"\right)"),
             "imag": (r"\mathrm{Im}\left(", r"\right)"), "angle": (r"\arg\left(", r"\right)"),
             **{f: (rf"\{f}\left(", r"\right)") for f in ["sin", "cos", "tan", "arctan", "sinh", "cosh", "tanh"]}}

# operator precedence, higher binds stronger
_prec = {ast.Add: 1, ast.Sub: 1, ast.Mult: 2, ast.Div: 2, ast.FloorDiv: 2, ast.Mod: 2, ast.Pow: 4}
_unary_prec = 3


def symbol_latex(name):
    if name in known_symbols:
        return known_symbols[name]

    # split into base and subscript: "tau_d" -> \tau_{d}, "a2" -> a_{2}
    match = re.fullmatch(r"([A-Za-z]+?)_?(\d+|(?<=_)\w+)?", name)
    base, sub = (match.group(1), match.group(2)) if match else (name, None)
    if base in greek:
        base = "\\" + base
    elif len(base) > 1:
        base = rf"\mathrm{{{base}}}"
    if sub:
        sub = "\\" + sub if sub in greek or sub == "infty" else sub.replace("_", r"\_")
        base += f"_{{{sub}}}"

    return base


def _number_latex(value):
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, complex):
        if value.real == 0:
            return "i" if value.imag == 1 else _number_latex(value.imag) + r"\,i"
        return rf"\left({_number_latex(value.real)} + {_number_latex(value.imag)}\,i\right)"

    text = f"{value:g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        exponent = str(int(exponent))
        return f"10^{{{exponent}}}" if mantissa == "1" else rf"{mantissa}\cdot10^{{{exponent}}}"

    return text


def _is_number(node):
    return isinstance(node, ast.Constant) or (isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant))


def _to_latex(node):
    # returns (latex, precedence)
    if isinstance(node, ast.Constant):
        prec = 2 if isinstance(node.value, complex) and node.value.real == 0 and node.value.imag != 1 else 5
        return _number_latex(node.value), prec

    if isinstance(node, ast.Name):
        return symbol_latex(node.id), 5

    if isinstance(node, ast.Attribute):  # np.pi -> pi
        return symbol_latex(node.attr), 5

    if isinstance(node, ast.UnaryOp):
        operand, op_prec = _to_latex(node.operand)
        if op_prec <= _unary_prec:
            operand = rf"\left({operand}\right)"
        sign = {ast.USub: "-", ast.UAdd: "+", ast.Not: r"\neg "}.get(type(node.op), "")
        return sign + operand, _unary_prec

    if isinstance(node, ast.BinOp):
        prec = _prec.get(type(node.op), 2)
        left, left_prec = _to_latex(node.left)
        right, right_prec = _to_latex(node.right)

        if isinstance(node.op, ast.Div):
            return rf"\frac{{{left}}}{{{right}}}", 5

        if isinstance(node.op, ast.Pow):
            if left_prec < 5 or isinstance(node.left, ast.Call):
                left = rf"\left({left}\right)"
            return f"{left}^{{{right}}}", prec

        # left operand needs parentheses if it binds weaker, right one also if it binds equally (a - (b + c))
        if left_prec < prec:
            left = rf"\left({left}\right)"
        if (right_prec < prec or (right_prec == prec and isinstance(node.op, (ast.Sub, ast.Mod, ast.FloorDiv)))
                or isinstance(node.right, ast.UnaryOp)):  # a - (-b), a (-b)
            right = rf"\left({right}\right)"

        if isinstance(node.op, ast.Mult):
            joiner = r"\cdot" if _is_number(node.right) else r"\,"
            return f"{left}{joiner}{right}", prec

        op = {ast.Add: "+", ast.Sub: "-", ast.FloorDiv: r"\,//\,", ast.Mod: r"\,\mathrm{mod}\,"}[type(node.op)]
        return f"{left} {op} {right}", prec

    if isinstance(node, ast.Call):
        func = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "f")
        args = ", ".join(_to_latex(arg)[0] for arg in node.args)
        if func in functions:
            start, end = functions[func]
            return f"{start}{args}{end}", 5
        name = func.replace("_", r"\_")
        return rf"\mathrm{{{name}}}\left({args}\right)", 5

    if isinstance(node, ast.Compare) or isinstance(node, ast.IfExp):
        raise ValueError("Comparisons and conditional expressions can't be converted to latex")

    raise ValueError(f"Unsupported syntax: {ast.dump(node)[:40]}")


def expression_to_latex(expression, lhs="y(f)"):
    tree = ast.parse(wrap_expression(expression), mode="eval")
    rhs, _ = _to_latex(tree.body)

    return f"{lhs} = {rhs}" if lhs else rhs


# latex symbols in expressions: \sigma_0, \omega_{p}, \varepsilon_\infty, \tau_{eff}, \gamma
_latex_symbol = re.compile(r"\\([A-Za-z]+)(?:_(?:\{\\?([A-Za-z0-9]+)\}|\\([A-Za-z]+)|([A-Za-z0-9])))?")


def latex_symbols_to_python(expression):
    # \sigma_0 -> sigma_0, \varepsilon_\infty -> varepsilon_infty (symbol_latex converts them back)
    def to_identifier(match):
        name, sub = match.group(1), match.group(2) or match.group(3) or match.group(4)
        return f"{name}_{sub}" if sub else name

    return _latex_symbol.sub(to_identifier, expression)


def wrap_expression(expression):
    # allows line breaks (and comments) anywhere in the expression and latex symbols as names
    return "(\n" + latex_symbols_to_python(expression.strip()) + "\n)"


greek_unicode = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ϵ", "varepsilon": "ε",
                 "zeta": "ζ", "eta": "η", "theta": "θ", "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ",
                 "nu": "ν", "xi": "ξ", "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "ϕ",
                 "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ",
                 "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ",
                 "Omega": "Ω", "infty": "∞"}
_subscript_digits = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def identifier_label(name):
    # display name for the parameter tables: sigma_0 -> σ₀, varepsilon_infty -> ε_∞, plain names -> ""
    base, _, sub = name.partition("_")
    if base not in greek_unicode:
        return ""
    label = greek_unicode[base]
    if sub.isdigit():
        label += sub.translate(_subscript_digits)
    elif sub:
        label += "_" + greek_unicode.get(sub, sub)

    return label
