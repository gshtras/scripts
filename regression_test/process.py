import pandas as pd

def main():
    with open('/projects/result_regression.log') as f:
        lines = f.readlines()

    try:
        df = pd.read_csv('/projects/regression.csv')
    except:
        df = pd.DataFrame(columns=[
            'date', 'model', 'batch', 'input_len', 'output_len', 'tp',
            'latency'
        ])
    date = lines[0].strip()
    found_pref = False
    for line in lines:
        if 'Performance' in line:
            found_pref = True
            continue
        if not found_pref:
            continue

        model, batch, input_len, output_len, tp, latency = line.split(',')
        batch = int(batch)
        input_len = int(input_len)
        output_len = int(output_len)
        tp = int(tp)
        latency = str(round(float(latency.split(' ')[2]), 4))

        new_df = pd.DataFrame({
            'date': [date],
            'model': [model],
            'batch': [batch],
            'input_len': [input_len],
            'output_len': [output_len],
            'tp': [tp],
            'latency': [latency]
        })
        if df[(df['date'] == date) & (df['model'] == model) &
              (df['batch'] == batch) & (df['input_len'] == input_len) &
              (df['output_len'] == output_len) & (df['tp'] == tp)].empty:
            df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(f'/projects/regression.csv', index=False)

    combos = df.loc[:, ["model", "batch", "input_len", "output_len", "tp"
                        ]].drop_duplicates().sort_values(by=[
                            "model", "batch", "input_len", "output_len", "tp"
                        ])
    with open("/projects/www-root/index.html", "w") as f:
        f.write(
            "<html><body><head><link rel='stylesheet' href='index.css'></head><table><tr><th>Model</th><th>Batch</th><th>Input Length</th><th>Output Length</th><th>TP</th>"
        )
        dates = df.loc[:, "date"].drop_duplicates()
        for i in range(len(dates)):
            f.write(f"<th>{dates.iloc[i]}</th>")
        f.write("</tr>")
        for i in range(len(combos)):
            combo = combos.iloc[i]
            matching_rows = df[(df['model'] == combo['model'])
                               & (df['batch'] == combo['batch']) &
                               (df['input_len'] == combo['input_len']) &
                               (df['output_len'] == combo['output_len']) &
                               (df['tp'] == combo['tp'])]
            f.write(
                f"<tr><td>{combo['model']}</td><td>{combo['batch']}</td><td>{combo['input_len']}</td><td>{combo['output_len']}</td><td>{combo['tp']}</td>"
            )
            last_latency = 0
            for date_itr in range(len(dates)):
                date = dates.iloc[date_itr]
                if matching_rows[matching_rows['date'] == date].empty:
                    f.write("<td></td>")
                    continue
                latency = float(matching_rows[matching_rows['date'] ==
                                              date].iloc[0]['latency'])
                if last_latency == 0:
                    classname = 'neutral'
                else:
                    ratio = latency / last_latency
                    if ratio > 1.1:
                        classname = 'bad'
                    elif ratio < 0.9:
                        classname = 'good'
                last_latency = latency
                f.write(f"<td class='{classname}'>{latency}</td>")
            f.write("</tr>")
        f.write("</table></body></html>")


if __name__ == '__main__':
    main()
