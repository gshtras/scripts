from enum import Enum
import pandas as pd
import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()
connection = psycopg2.connect(database=os.getenv("DASHBOARD_DATABASE"),
                              user=os.getenv("DASHBOARD_USER"),
                              password=os.getenv("DASHBOARD_PASSWORD"),
                              host=os.getenv("DASHBOARD_HOST"),
                              port=os.getenv("DASHBOARD_PORT"))

cursor = connection.cursor()


def fix_model_name(model: str):
    return model.strip('/').replace('/', '_')


def find_projects():
    projects_path = os.path.join(os.path.expanduser("~"), "Projects")
    if os.path.exists(projects_path):
        return projects_path
    projects_path = os.path.join(os.path.abspath(os.sep), "projects")
    if os.path.exists(projects_path):
        return projects_path
    raise Exception("Projects path not found")


class ParseState(Enum):
    ExpectModel = 0
    ExpectResult = 1


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
    version = lines[1].strip().split('Version: ')[-1]
    found_pref = False
    found_correctness = False
    found_p3l = False
    correctness_output = ""
    p3l_output = ""
    parse_state = ParseState.ExpectModel
    model_str = ""

    for line in lines:
        if "===Vision===" in line:
            continue
        if '===Correctness===' in line:
            found_correctness = True
            parse_state = ParseState.ExpectModel
            continue
        if '===Performance===' in line:
            found_pref = True
            continue
        if "===P3L===" in line:
            found_p3l = True
            parse_state = ParseState.ExpectModel
            continue
        if found_p3l:
            p3l_output += line.replace("\n", "<br/>")
            if parse_state == ParseState.ExpectModel:
                if "Integral Cross-Entropy=" not in line:
                    parse_state = ParseState.ExpectResult
                    model_str = line.strip()
            elif parse_state == ParseState.ExpectResult:
                parse_state = ParseState.ExpectModel
                if "Integral Cross-Entropy=" in line:
                    cursor.execute(
                        'INSERT INTO "Model" (name, path) VALUES (%s, %s) ON CONFLICT DO NOTHING',
                        ((fix_model_name(model_str.split(',')[0]),
                          model_str.split(',')[0])))
                    cursor.execute(
                        '''
                        INSERT INTO "P3LConfig" ("modelName", "contextLen", "sampleSize", "patchSize", dtype)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                        ''', (fix_model_name(model_str.split(',')[0]), model_str.split(',')[1],
                              model_str.split(',')[2], model_str.split(',')[3],
                              'auto'))
                    cursor.execute('COMMIT')
                    cursor.execute(
                        'SELECT id from "P3LConfig" WHERE "modelName"=%s AND "contextLen"=%s AND "sampleSize"=%s AND "patchSize"=%s AND dtype=%s',
                        (fix_model_name(model_str.split(',')[0]),
                         model_str.split(',')[1], model_str.split(',')[2],
                         model_str.split(',')[3], 'auto'))
                    data = cursor.fetchone()
                    cursor.execute(
                        '''
                        INSERT INTO "P3LResult" ("p3lConfigId", "createdAt", "P3L", "vllmVersion")
                          VALUES (%s, %s, %s, %s)
                          ON CONFLICT DO NOTHING
                                   ''',
                        (data[0], date, line.split('PPL=')[1], version))
                    cursor.execute('COMMIT')
            continue
        if found_correctness and not found_pref:
            correctness_output += line.replace("\n", "</p><p>")
            if parse_state == ParseState.ExpectModel:
                if "Generated:" not in line:
                    parse_state = ParseState.ExpectResult
                    model_str = line.strip()
            elif parse_state == ParseState.ExpectResult:
                parse_state = ParseState.ExpectModel
                if "Generated: " in line:
                    cursor.execute(
                        'INSERT INTO "Model" (name, path) VALUES (%s, %s) ON CONFLICT DO NOTHING',
                        ((fix_model_name(model_str.split(',')[0]),
                          model_str.split(',')[0])))
                    cursor.execute(
                        '''
                        INSERT INTO "CorrectnessConfig" ("modelName", dtype) VALUES (%s, %s) ON CONFLICT DO NOTHING
                                ''', (fix_model_name(
                            model_str.split(',')[0]), model_str.split(',')[1]))
                    cursor.execute('COMMIT')
                    cursor.execute(
                        'SELECT id from "CorrectnessConfig" WHERE "modelName"=%s AND dtype=%s',
                        (fix_model_name(
                            model_str.split(',')[0]), model_str.split(',')[1]))
                    data = cursor.fetchone()
                    cursor.execute(
                        '''
                        INSERT INTO "CorrectnessResult" ("correctnessConfigId", "createdAt", generated, "vllmVersion") VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING
                                   ''',
                        (data[0], date, line.split('Generated: ')[1], version))
                    cursor.execute('COMMIT')
            continue
        if not found_pref:
            continue

        parts = line.split(',')
        if len(parts) == 6:
            model, batch, input_len, output_len, tp, latency = line.split(',')
            dtype = 'bfloat16'
        elif len(parts) == 7:
            model, batch, input_len, output_len, tp, dtype, latency = line.split(
                ',')
        else:
            continue
        batch = int(batch)
        input_len = int(input_len)
        output_len = int(output_len)
        tp = int(tp)
        try:
            latency = str(round(float(latency.strip()), 4))
        except:
            continue

        cursor.execute(
            'INSERT INTO "Model" (name, path) VALUES (%s, %s) ON CONFLICT DO NOTHING',
            (fix_model_name(model), model))
        cursor.execute(
            'INSERT INTO "PerformanceConfig" (dtype, "batchSize", "inputLength", "outputLength", tp, "modelName") VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING',
            (dtype, batch, input_len, output_len, tp, fix_model_name(model)))
        cursor.execute('COMMIT')
        cursor.execute(
            'SELECT id from "PerformanceConfig" WHERE dtype=%s AND "batchSize"=%s AND "inputLength"=%s AND "outputLength"=%s AND tp=%s AND "modelName"=%s',
            (dtype, batch, input_len, output_len, tp, fix_model_name(model)))
        data = cursor.fetchone()
        cursor.execute(
            '''
            INSERT INTO "PerformanceResult" 
            ("createdAt", "vllmVersion", "performanceConfigId", latency) 
            VALUES 
            (%s, %s, %s, %s) 
            ON CONFLICT DO NOTHING
            ''', (date, version, data[0], latency))
        cursor.execute('COMMIT')

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

    combos = df.loc[:, [
        "model", "batch", "input_len", "output_len", "tp", "dtype"
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
                               (df['tp'] == combo['tp']) &
                               (df['dtype'] == combo['dtype'])]
            if len(matching_rows) == 0:
                continue
            model_row = f"<tr><td>{combo['model']}</td><td class='num'>{combo['batch']}</td><td class='num'>{combo['input_len']}</td><td class='num'>{combo['output_len']}</td><td class='num'>{combo['tp']}</td><td>{combo['dtype']}</td>"
            has_data = False
            last_latency = 0
            avg_latency = 0
            num_entries = 0
            for date_itr in range(len(dates)):
                date = dates.iloc[date_itr]
                if matching_rows[matching_rows['date'] == date].empty:
                    model_row += "<td></td>"
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
                model_row += f"<td class='num{classname}'>{latency}</td>"
                has_data = True
            model_row += "</tr>"
            if has_data:
                f.write(model_row)
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
                                        "archive")))[-15:]:
            if file.endswith(".html"):
                f.write(
                    f"<a href='archive/{file}'>{file.replace('.html','')}</a><br>"
                )
        archive_f.write(f"</body></html>")


if __name__ == '__main__':
    main()
