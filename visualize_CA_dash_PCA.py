import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import CSGridMLMDataset
from tqdm import tqdm
import numpy as np
# from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
import plotly.express as px
import re
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
from generate_utils import load_SEFiLMModel, generate_files_with_nucleus, get_SE_embeddings_for_sequence
import os
import pickle
from tqdm import tqdm

# global
df = None
data_all = None
pca = None

device_name = 'cuda:0'
val_dir = '/mnt/ssd2/maximos/data/mel_harm_other_CA/nottingham_test'
jazz_dir = '/mnt/ssd2/maximos/data/gjt_melodies/gjt_CA_test'

tokenizer = CSGridMLMTokenizer(
    fixed_length=80,
    quantization='4th',
    intertwine_bar_info=True,
    trim_start=False,
    use_pc_roll=True,
    use_full_range_melody=False
)

loss_scheme = 'fhl'
model = load_SEFiLMModel(
    tokenizer,
    loss_scheme,
    device_name,
    d_model=512
)

val_files = [f for f in os.listdir(val_dir) if f.endswith('.mid') or f.endswith('.midi')]
jazz_files = [f for f in os.listdir(jazz_dir) if f.endswith('.mxl') or f.endswith('.xml')]

val_data = []
jazz_data = []

print('making val data struct')
for f in tqdm(val_files):
    tmp_struct = {}
    tmp_path = os.path.join( val_dir, f )
    tmp_struct['path'] = tmp_path
    encoded = tokenizer.encode( tmp_path )
    tmp_struct['pianoroll'] = encoded['pianoroll']
    tmp_struct['harmony_ids'] = encoded['harmony_ids']
    latent = get_SE_embeddings_for_sequence(model, encoded['pianoroll'], encoded['harmony_ids']).detach().cpu().numpy()
    tmp_struct['latent'] = latent
    tmp_struct['name'] = f.split('.')[0]
    val_data.append(tmp_struct)
print('making jazz data struct')
for f in tqdm(jazz_files):
    tmp_struct = {}
    tmp_path = os.path.join( jazz_dir, f )
    tmp_struct['path'] = tmp_path
    encoded = tokenizer.encode( tmp_path )
    tmp_struct['pianoroll'] = encoded['pianoroll']
    tmp_struct['harmony_ids'] = encoded['harmony_ids']
    latent = get_SE_embeddings_for_sequence(model, encoded['pianoroll'], encoded['harmony_ids']).detach().cpu().numpy()
    tmp_struct['latent'] = latent
    tmp_struct['name'] = f.split('.')[0]
    jazz_data.append(tmp_struct)

mxl_folder_out = 'examples_musicXML/' + loss_scheme + '/'
midi_folder_out = 'examples_MIDI/' + loss_scheme + '/'
os.makedirs(mxl_folder_out, exist_ok=True)
os.makedirs(midi_folder_out, exist_ok=True)

custom_colors = ["#42b41f", "#0e2eff", '#d62728']  # blue, orange, red
symbol_map = {
    '0': 'circle',
    '1': 'circle',
    '2': 'square'  # for harmonized points
}
size_map = {
    '0': 10,
    '1': 10,
    '2': 15  # larger for harmonized points
}

if device_name == 'cpu':
        device = torch.device('cpu')
else:
    if torch.cuda.is_available():
        device = torch.device(device_name)
    else:
        print('Selected device not available: ' + device_name)

def condenced_str_from_token_ids(inp_ids, tokenizer):
    # for computing features
    tmp_str = ''
    tmp_count = 0
    prev_id = -1
    num_chords = 0
    for t in inp_ids:
        if t != tokenizer.bar_token_id:
            if prev_id == t:
                tmp_count += 1
            else:
                if prev_id != -1:
                    chord_token = tokenizer.ids_to_tokens[prev_id]
                    tmp_str += f'{tmp_count}x{chord_token}'
                    num_chords += 1
                    if num_chords == 4:
                        tmp_str += '\n'
                        num_chords = 0
                    else:
                        tmp_str += '_'
                tmp_count = 1
                prev_id = t
    chord_token = tokenizer.ids_to_tokens[prev_id]
    tmp_str += f'{tmp_count}x{chord_token}'
    return tmp_str
# end condenced_str_from_token_ids


def extract_chord_labels(text):
    labels = []
    for chunk in re.split(r'[_\n]+', text):
        if not chunk:
            continue
        parts = chunk.split('x', 1)
        labels.append(parts[-1])
    return labels


def make_colored_chord_line(line, style_fn):
    children = []
    tokens = line.split('_')
    for i, chunk in enumerate(tokens):
        if i > 0:
            children.append(html.Span('_', style={'margin': '0 4px'}))
        if not chunk:
            continue
        label = chunk.split('x', 1)[-1]
        color = style_fn(label)
        children.append(html.Span(
            chunk,
            style={
                'backgroundColor': color,
                'padding': '2px 6px',
                'borderRadius': '4px',
                'marginRight': '4px',
                'display': 'inline-block',
                'whiteSpace': 'nowrap'
            }
        ))
    return children


def render_colored_token_block(token_lines, style_fn):
    if not token_lines:
        return html.Div()
    children = [html.Div(token_lines[0], style={'fontWeight': 'bold', 'marginBottom': '4px'})]
    for line in token_lines[1:]:
        children.append(html.Div(
            make_colored_chord_line(line, style_fn),
            style={'display': 'inline-flex', 'flexWrap': 'wrap', 'alignItems': 'center', 'marginBottom': '4px'}
        ))
    return html.Div(children)


def render_colored_chord_lines(token_lines, style_fn):
    children = []
    for line in token_lines:
        if not line.strip():
            continue
        children.append(html.Div(
            make_colored_chord_line(line, style_fn),
            style={'display': 'inline-flex', 'flexWrap': 'wrap', 'alignItems': 'center', 'marginBottom': '4px'}
        ))
    return html.Div(children)


def render_harmonized_block(token_lines, melody_labels, guide_labels):
    common_labels = melody_labels & guide_labels
    melody_only = melody_labels - guide_labels
    guide_only = guide_labels - melody_labels

    def style_fn(label):
        if label in common_labels:
            return '#d3d3d3'
        if label in melody_only:
            return '#b8e994'
        if label in guide_only:
            return '#ffb3b3'
        return '#fff3b0'

    return render_colored_chord_lines(token_lines, style_fn)


def render_melody_guide_blocks(melody_tokens, guide_tokens):
    melody_labels = set(extract_chord_labels('\n'.join(melody_tokens[1:])))
    guide_labels = set(extract_chord_labels('\n'.join(guide_tokens[1:])))
    common_labels = melody_labels & guide_labels

    def melody_style(label):
        return '#d3d3d3' if label in common_labels else '#b8e994'

    def guide_style(label):
        return '#d3d3d3' if label in common_labels else '#ffb3b3'

    return (
        render_colored_token_block(melody_tokens, melody_style),
        render_colored_token_block(guide_tokens, guide_style),
        melody_labels,
        guide_labels
    )


def apply_pca(model, tokenizer, val_dataset, jazz_dataset):
    global pca
    print('FUN apply_pca')
    zs = []
    z_idxs = []
    z_tokens = []
    feats = []
    global data_all, df
    data_all = []
    for d in tqdm(val_dataset):
        data_all.append(d)
        feats.append(d['latent'])
        tmp_str = condenced_str_from_token_ids(d['harmony_ids'], tokenizer)
        z_idxs.append(0)
        z_tokens.append(d['name'] + '\n' + tmp_str)
    for d in tqdm(jazz_dataset):
        data_all.append(d)
        feats.append(d['latent'])
        z_idxs.append(1)
        tmp_str = condenced_str_from_token_ids(d['harmony_ids'], tokenizer)
        z_tokens.append(d['name'] + '\n' + tmp_str)

    # z_np = np.array( zs )
    feats_np = np.array(feats)

    # pca = PCA(n_components=2)
    pca = KernelPCA(n_components=2, kernel='cosine')
    # y = pca.fit_transform( z_np )
    y = pca.fit_transform( feats_np )
    
    # Combine into a DataFrame for easy Plotly integration
    df = pd.DataFrame({
        'x': y[:, 0],
        'y': y[:, 1],
        'class': z_idxs,
        'token': z_tokens
    })

    df['hover_text'] = df['token'].str.replace('\n', '<br>')
    df['class_str'] = df['class'].astype(str)
    df['symbol'] = df['class_str'].map(symbol_map)
    df['size'] = df['class_str'].map(size_map)
# end apply_pca

def make_figure(selected):
    print('FUN make_figure')
    global df
    # Create interactive scatter plot
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='class_str',
        symbol='symbol',
        custom_data=['hover_text'],
        # size='size',
        # size_max=10,
        hover_data=None,
        color_discrete_sequence=custom_colors,
        symbol_sequence=list(symbol_map.values())
    )

    fig.update_layout(
        xaxis=dict(scaleanchor='y', scaleratio=1),  # Equal aspect ratio
    )

    fig.update_traces(
        hovertemplate='%{customdata[0]}<extra></extra>'
    )
    print(selected)
    if selected:
        if selected['melody'] is not None:
            row = df.iloc[selected['melody']]
            fig.add_scatter(
                x=[row['x']],
                y=[row['y']],
                mode='markers',
                marker=dict(    
                    color='green',
                    size=16,
                    symbol='star',
                    line=dict(color='black', width=2),
                    opacity=1.0
                ),
                name='Melody',
                showlegend=True
            )

        if selected['guide'] is not None:
            row = df.iloc[selected['guide']]
            fig.add_scatter(
                x=[row['x']],
                y=[row['y']],
                mode='markers',
                marker=dict(
                    color='red',
                    size=16,
                    symbol='star',
                    line=dict(color='black', width=2),
                    opacity=1.0
                ),
                name='Guide',
                showlegend=True
            )
    return fig
# end make_figure

apply_pca(model, tokenizer, val_data, jazz_data)

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='scatter-plot', figure=make_figure(None)),
    dcc.Store(id='selected-points', data={'melody': None, 'guide': None}),
    html.Div([
        # Left side: Selections info
        html.Div(id='click-output', style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        # Right side: Harmonize button + result
        html.Div([
            html.Button("Harmonize", id='harmonize-button', n_clicks=0),
            html.Div(id='harmonize-output', style={'marginTop': '10px'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '20px'})
    ], style={'display': 'flex', 'justifyContent': 'space-between'})
])

@app.callback(
    Output('scatter-plot', 'figure', allow_duplicate=True),
    Output('click-output', 'children'),
    Output('selected-points', 'data'),
    Input('scatter-plot', 'clickData'),
    State('selected-points', 'data'),
    prevent_initial_call=True
)

def handle_click(clickData, selected):
    if clickData is None:
        return dash.no_update
    point = clickData['points'][0]
    point_index = point['pointIndex']
    curve_number = point['curveNumber']

    # Find the global index in the DataFrame
    # Get trace class based on curveNumber
    clicked_class = df['class'].unique()[curve_number]

    # Get the row indices in df for that class
    matching_indices = df[df['class'] == clicked_class].index.to_list()
    
    # Use pointIndex within that group to get the global index
    idx = matching_indices[point_index]

    if selected:
        if selected['melody'] is None or selected['melody'] == idx:
            selected['melody'] = idx
        elif selected['guide'] is None or selected['guide'] == idx:
            selected['guide'] = idx
        else:
            # Rotate selection
            selected['melody'], selected['guide'] = selected['guide'], idx

    token1 = df.iloc[selected['melody']]['token'].split('\n') if selected['melody'] is not None else ["None"]
    token2 = df.iloc[selected['guide']]['token'].split('\n') if selected['guide'] is not None else ["None"]
    melody_block, guide_block, _, _ = render_melody_guide_blocks(token1, token2)

    text = html.Div([
        html.Strong("Melody:"),
        melody_block,
        html.Br(),
        html.Strong("Guide:"),
        guide_block,
    ])

    return make_figure(selected), text, selected

@app.callback(
    Output('scatter-plot', 'figure', allow_duplicate=True),
    Output('harmonize-output', 'children'),
    Input('harmonize-button', 'n_clicks'),
    State('selected-points', 'data'),
    prevent_initial_call=True,
)
def run_harmonization(n_clicks, selected):
    global pca, df
    if not selected or selected['melody'] is None or selected['guide'] is None:
        return "Please select both a melody and a guide first."
    input_data = data_all[selected['melody']]
    guide_data = data_all[selected['guide']]
    g = generate_files_with_nucleus(
        model,
        tokenizer,
        input_data['path'],
        None,
        midi_folder_out,
        name_suffix='test',
        guidance_f_path=guide_data['path'],
        guidance_vec=None,
        use_constraints=False,
        intertwine_bar_info=True, # no bar default
        normalize_tonality=False,
        temperature=0.3,
        p=0.9,
        unmasking_order='certain',
        create_gen = True,
        create_real = True,
        create_guide = True
    )
    gen_output_tokens = g['gen_output_tokens']
    # text to present to html
    gen_output_html = condenced_str_from_token_ids(g['gen_output_token_ids'][0].tolist(), tokenizer)
    gen_output_html = gen_output_html.split('\n')
    melody_labels = set(extract_chord_labels(df.iloc[selected['melody']]['token'].split('\n', 1)[1] if '\n' in df.iloc[selected['melody']]['token'] else ''))
    guide_labels = set(extract_chord_labels(df.iloc[selected['guide']]['token'].split('\n', 1)[1] if '\n' in df.iloc[selected['guide']]['token'] else ''))
    txt = html.Div([
        html.Strong("Harmonized:"),
        render_harmonized_block(gen_output_html, melody_labels, guide_labels)
    ])
    hover_txt = 'Harmonized:<br>' + '<br>'.join(gen_output_html)
    # embedding to apply pca transformation to
    print('guide-z: ', F.cosine_similarity(
        g['hidden'].detach().cpu(),
        torch.FloatTensor(guide_data['latent']), dim=-1
    ))
    print('input-z: ', F.cosine_similarity(
        g['hidden'].detach().cpu(),
        torch.FloatTensor(input_data['latent']), dim=-1
    ))
    # appy pca
    # z_pca = pca.transform([z])[0]
    z_pca = pca.transform([g['hidden'].detach().cpu().numpy()])[0]
    # append new point to df
    new_point = {
        'x': z_pca[0],
        'y': z_pca[1],
        'class': 2,
        'token': gen_output_tokens,
        'hover_text': hover_txt,
        'class_str': '2',
        'symbol': symbol_map['2'],
        'size': size_map['2']
    }
    print(z_pca)
    guide_z = [ df.iloc[selected['guide']]['x'] , df.iloc[selected['guide']]['y'] ]
    input_z = [ df.iloc[selected['melody']]['x'] , df.iloc[selected['melody']]['y'] ]
    print('PCA guide-z', F.cosine_similarity(torch.FloatTensor(z_pca), torch.FloatTensor(guide_z), dim=-1))
    print('PCA input-z', F.cosine_similarity(torch.FloatTensor(z_pca), torch.FloatTensor(input_z), dim=-1))
    df = pd.concat([df, pd.DataFrame([new_point])], ignore_index=True)
    return make_figure(selected), txt

if __name__ == '__main__':
    print('FUN main')
    app.run(debug=True, port=3052)