from pathlib import Path
import os

# This builds out a html file
# Updating this might be pain

html = """
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
</head>
<body>

<h2>Interactive Cost Function</h2>

<label for="a">Amplitude:</label>
<input type="number" id="a" min="1" max="10" value="1" step="0.1">

<label for="b">Power:</label>
<input type="number" id="b" min="0" max="1" value="1" step="0.1">

<p>
Amplitude value:
<span id="aval">1</span>
</p>

<p>
Power value:
<span id="bval">1</span>
</p>

<div id="plot"></div>

<script>

function updatePlot() {

    const a = parseFloat(document.getElementById("a").value);
    const b = parseFloat(document.getElementById("b").value);
    document.getElementById("aval").innerText = a;
    document.getElementById("bval").innerText = b;

    let x = [];
    let y = [];

    for (let i = 0; i < 300; i++) {

        let xi = i * 0.05;

        x.push(xi);

        y.push(a * Math.pow(xi, b));
    }

    Plotly.newPlot(
        'plot',
        [{
            x: x,
            y: y,
            mode: 'lines',
            name: 'y = a * x^b'
        }],
        {
            title: `y = ${a} * x^${b}`,
            margin: {t: 50}
        }
    );
}

document
    .getElementById("a")
    .addEventListener("input", updatePlot);

document
    .getElementById("b")
    .addEventListener("input", updatePlot);

updatePlot();

</script>

</body>
</html>
"""

cwd = os.getcwd()
print(f"Current working directory: {cwd}")
path = os.path.join(cwd, "interactive_costing/cost_functions")

output = Path(path) / "interactive_cost_function.html"
output.write_text(html)

print(f"Wrote {output.resolve()}")
