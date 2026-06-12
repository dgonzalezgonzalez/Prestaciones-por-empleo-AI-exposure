const fs = require("fs");
const path = require("path");
const readline = require("readline");

const projectRoot = path.resolve(__dirname, "..", "..");
const panelPath = path.join(projectRoot, "data", "processed", "analysis_panel.csv");
const outPath = path.join(projectRoot, "output", "tables", "referee2_node_data_check.json");
const exposureName = "observed_exposure_cosine_nearest";

function parseCsvLine(line) {
  const fields = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === "," && !inQuotes) {
      fields.push(current);
      current = "";
    } else {
      current += ch;
    }
  }
  fields.push(current);
  return fields;
}

function quantile(values, q) {
  const sorted = values.slice().sort((a, b) => a - b);
  const pos = (sorted.length - 1) * q;
  const lower = Math.floor(pos);
  const upper = Math.ceil(pos);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (pos - lower);
}

async function main() {
  const rl = readline.createInterface({
    input: fs.createReadStream(panelPath),
    crlfDelay: Infinity,
  });

  let header = null;
  let idx = {};
  let rows = 0;
  const occupations = new Set();
  const periods = new Set();
  const exposures = [];
  for await (const line of rl) {
    if (header === null) {
      header = parseCsvLine(line);
      header.forEach((name, i) => {
        idx[name] = i;
      });
      continue;
    }
    if (!line.trim()) continue;
    const fields = parseCsvLine(line);
    rows += 1;
    occupations.add(fields[idx.cno4]);
    periods.add(fields[idx.period]);
    exposures.push(Number(fields[idx[exposureName]]));
  }

  const p75 = quantile(exposures, 0.75);
  const result = {
    audit: "referee2_node_data_check",
    rows,
    occupations: occupations.size,
    periods: periods.size,
    threshold_p75: p75,
    high_rows: exposures.filter((x) => x > p75).length,
    zero_rows: exposures.filter((x) => x === 0).length,
    note: "Pure Node.js CSV parse of processed panel; verifies counts and treatment threshold independently of pandas.",
  };
  fs.writeFileSync(outPath, JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
