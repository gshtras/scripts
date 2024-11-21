import pandas as pd
import os


def find_projects():
    projects_path = os.path.join(os.path.expanduser("~"), "Projects")
    if os.path.exists(projects_path):
        return projects_path
    projects_path = os.path.join(os.path.abspath(os.sep), "projects")
    if os.path.exists(projects_path):
        return projects_path
    raise Exception("Projects path not found")


def main():
    projects_path = find_projects()
    with open(os.path.join(projects_path, 'result_regression.log')) as f:
        lines = f.readlines()

    try:
        df = pd.read_csv(os.path.join(projects_path, 'regression.csv'))
        if 'dtype' not in df:
            df['dtype'] = 'bfloat16'
    except:
        df = pd.DataFrame(columns=[
            'date', 'model', 'batch', 'input_len', 'output_len', 'tp', 'dtype',
            'latency'
        ])
    date = lines[0].strip()
    version = lines[1].strip()
    found_pref = False
    found_correctness = False
    found_p3l = False
    correctness_output = ""
    p3l_output = ""
    for line in lines:
        if 'Correctness' in line:
            found_correctness = True
            continue
        if 'Performance' in line:
            found_pref = True
            continue
        if "P3L" in line:
            found_p3l = True
            continue
        if found_p3l:
            p3l_output += line.replace("\n", "<br/>")
            continue
        if found_correctness and not found_pref:
            correctness_output += line.replace("\n", "</p><p>")
        if not found_pref:
            continue

        try:
            model, batch, input_len, output_len, tp, dtype, latency = line.split(
                ',')
        except:
            model, batch, input_len, output_len, tp, latency = line.split(',')
            dtype = 'bfloat16'
        batch = int(batch)
        input_len = int(input_len)
        output_len = int(output_len)
        tp = int(tp)
        try:
            latency = str(round(float(latency.strip()), 4))
        except:
            continue

        new_df = pd.DataFrame({
            'date': [date],
            'model': [model],
            'batch': [batch],
            'input_len': [input_len],
            'output_len': [output_len],
            'tp': [tp],
            'dtype': [dtype],
            'latency': [latency]
        })
        if df[(df['date'] == date) & (df['model'] == model) &
              (df['batch'] == batch) & (df['input_len'] == input_len) &
              (df['output_len'] == output_len) & (df['tp'] == tp) &
              (df['dtype'] == dtype)].empty:
            df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(os.path.join(projects_path, 'regression.csv'), index=False)

    combos = df.loc[:, ["model", "batch", "input_len", "output_len", "tp", "dtype"
                        ]].drop_duplicates().sort_values(by=[
                            "dtype",
                            "model",
                            "tp",
                            "batch",
                            "input_len",
                            "output_len",
                        ])
    with open(os.path.join(projects_path, "www-root", "index.html"),
              "w") as f, open(
                  os.path.join(projects_path, "www-root", "archive",
                               f"{date}.html"), "w") as archive_f:
        archive_f.write(
            "<html><head><link rel='stylesheet' href='index.css'></head><body>"
        )
        f.write("""
            <html><head><link rel='stylesheet' href='index.css'><script src='table.js'></script></head><body><h1>Performance results - Latency (s)</h1>
            <table class='sortable'><thead>
            <tr>
            <th aria-sort='ascending'><button>Model<span aria-hidden="true"></span></button></th>
            <th class="num"><button>Batch<span aria-hidden="true"></span></button></th>
            <th class="num"><button>Input Length<span aria-hidden="true"></span></button></th>
            <th class="num"><button>Output Length<span aria-hidden="true"></span></button></th>
            <th class="num"><button>TP<span aria-hidden="true"></span></button></th>
            <th><button>dtype<span aria-hidden="true"></span></button></th>
            """)
        dates = df.loc[:, "date"].drop_duplicates()[-15:]
        for i in range(len(dates)):
            f.write(f"<th class='no-sort'>{dates.iloc[i]}</th>")
        f.write("</tr><tbody>")
        for i in range(len(combos)):
            combo = combos.iloc[i]
            matching_rows = df[(df['model'] == combo['model'])
                               & (df['batch'] == combo['batch']) &
                               (df['input_len'] == combo['input_len']) &
                               (df['output_len'] == combo['output_len']) &
                               (df['tp'] == combo['tp']) & (df['dtype'] == combo['dtype'])]
            if len(matching_rows) == 0:
                continue
            f.write(
                f"<tr><td>{combo['model']}</td><td class='num'>{combo['batch']}</td><td class='num'>{combo['input_len']}</td><td class='num'>{combo['output_len']}</td><td class='num'>{combo['tp']}</td><td>{combo['dtype']}</td>"
            )
            last_latency = 0
            avg_latency = 0
            num_entries = 0
            for date_itr in range(len(dates)):
                date = dates.iloc[date_itr]
                if matching_rows[matching_rows['date'] == date].empty:
                    f.write("<td></td>")
                    continue
                latency = float(matching_rows[matching_rows['date'] ==
                                              date].iloc[0]['latency'])
                avg_latency = (
                    (avg_latency * num_entries) + latency) / (num_entries + 1)
                num_entries += 1
                classname = ''
                last_ratio = 1.0
                avg_ratio = 1.0
                if avg_latency > 0:
                    avg_ratio = latency / avg_latency
                if last_latency > 0:
                    last_ratio = latency / last_latency
                if (last_ratio > 1.1 and avg_ratio
                        > 1.1) or last_ratio > 2 or avg_ratio > 2:
                    classname = ' bad'
                elif (last_ratio < 0.9 and avg_ratio
                      < 0.9) or last_ratio < 0.5 or avg_ratio < 0.5:
                    classname = ' good'
                last_latency = latency
                f.write(f"<td class='num{classname}'>{latency}</td>")
            f.write("</tr>")
        f.write("</tbody></table>")
        f.write(f"<p>Version: {version}</p>")
        archive_f.write(f"<p>Version: {version}</p>")
        f.write(
            f"<h1>Correctness results on {date}</h1><p>{correctness_output}</p>"
        )
        f.write(f"<h1>P3L results on {date}</h1><p>{p3l_output}</p>")
        archive_f.write(
            f"<h1>Correctness results on {date}</h1><p>{correctness_output}</p>"
        )
        archive_f.write(f"<h1>P3L results on {date}</h1><p>{p3l_output}</p>")
        f.write("</body></html>")
        f.write("<h1>Archive</h1>")
        for file in sorted(
                os.listdir(os.path.join(projects_path, "www-root",
                                        "archive"))):
            if file.endswith(".html"):
                f.write(
                    f"<a href='archive/{file}'>{file.replace('.html','')}</a><br>"
                )
        archive_f.write(f"</body></html>")


if __name__ == '__main__':
    main()
