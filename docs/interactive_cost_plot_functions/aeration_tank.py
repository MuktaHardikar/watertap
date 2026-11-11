from pathlib import Path

# This builds out a html file

html = """
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
</head>
<body>

<h2>Cost Function</h2>

<label for="a">A:</label>
<input type="number" id="a" min="0" max="5000" value="1000" step="0.1">

<label for="b">B:</label>
<input type="number" id="b" min="0" max="1" value="0.5" step="0.1">

<div id="plot"></div>

<script>

function updatePlot() {

    const a = parseFloat(document.getElementById("a").value);
    const b = parseFloat(document.getElementById("b").value);

    let x = [];
    let y = [];

    for (let i = 0; i < 300; i++) {

        let xi = i;

        x.push(xi);
        y.push(a * Math.pow(xi, b));
    }

    Plotly.newPlot(
        'plot',
        [
            {
                x: x,
                y: y,
                mode: 'lines',
                name: `C_{cap,tot} = ${a} * V^${b}`
            },
            {
                x: x,
                y: x.map(xi => 1114*Math.pow(xi, 0.6)),
                mode: 'lines',
                name: 'Reference',
                line: { dash: 'dash', color: 'red' }
            }
        ],
        {
            margin: {t: 50},
            xaxis: { title: 'Volume (V)' },
            yaxis: { title: 'Capital Cost ($)'},
            title: 'Capital Cost vs Volume for Aeration Tank'
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

# Get parent directory of this file
parent_dir = Path(__file__).parent.parent
path = parent_dir / "_static/interactive_cost_plots/aeration_tank.html"
output = Path(path)
output.write_text(html)

print(f"Wrote {output.resolve()}")
