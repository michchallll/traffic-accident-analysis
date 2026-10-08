#!/usr/bin/env python3.12
# coding=utf-8

from matplotlib import dates, pyplot as plt
import pandas as pd
import seaborn as sns
import zipfile
import os

# Ukol 1: nacteni dat ze ZIP souboru
def load_data(filename: str, ds: str) -> pd.DataFrame:
    """
    Načte data ze složek 2023 a 2024 v ZIP archivu a spojí je do jednoho DataFrame.

    Tato funkce otevře ZIP soubor, prozkoumá složky obsahující data pro rok 2023 a 2024,
    načte tabulky z XLS souboru uvnitř každé složky, odstraní nepojmenované sloupce a
    spojí všechny tabulky do jednoho DataFrame.

    Args:
        filename (str): Cesta k ZIP souboru obsahujícímu XLS soubory.
        ds (str): Název XLS souboru bez prefixu "I". Používá se pro sestavení názvu souboru.

    Returns:
        pd.DataFrame: Spojený DataFrame obsahující data z obou složek (2023, 2024).
    """
    # Název souboru v jednotlivých složkách
    xls_filename = f"I{ds}.xls"
    combined_df = pd.DataFrame()

    with zipfile.ZipFile(filename, 'r') as zf:
        # Získání seznamu všech cest v ZIP souboru
        file_list = zf.namelist()
        # Získání unikátních názvů složek
        folders = set(os.path.dirname(path).split('/')[0] for path in file_list if path.endswith(xls_filename))
        
        # Iterace přes složky 2023 a 2024
        for folder in folders:
            # Sestavení cesty k souboru uvnitř ZIP archivu
            path = f"{folder}/{xls_filename}"
            with zf.open(path) as file:
                # Načíst data z HTML souboru
                tables = pd.read_html(file)  # Vrací list
                df = tables[0]  # Vyber první tabulku jako DataFrame
                # Odstranit nepojmenované sloupce
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                # Přidat data do výsledného DataFrame
                combined_df = pd.concat([combined_df, df], ignore_index=True)

    return combined_df

# Ukol 2: zpracovani dat
def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Čistí a formátuje DataFrame dle specifikace.

    Tato funkce provádí několik úprav na vstupním DataFrame, včetně:
    - Převodu sloupce `p2a` na formát data.
    - Mapování sloupce `p4a` na regiony podle definovaného slovníku.
    - Odstranění duplikátů na základě sloupce `p1`.
    - Možnost zobrazení velikosti datového rámce po úpravách, pokud je aktivní parametr `verbose`.

    Args:
        df (pd.DataFrame): Vstupní DataFrame, který bude upraven.
        verbose (bool): Pokud True, vypíše velikost datového rámce po úpravách. Výchozí je False.

    Returns:
        pd.DataFrame: Upravený a vyčištěný DataFrame.
    """
    # Vytvoření nového DataFrame
    df_cleaned = df.copy()

    # Převod sloupce `p2a` na formát data (nový sloupec `date`)
    df_cleaned['date'] = pd.to_datetime(df_cleaned['p2a'], format='%d.%m.%Y')

    # Vytvoření sloupce `region` dle mapování
    region_map = {
        0: "PHA", 1: "STC", 2: "JHC", 3: "PLK", 4: "ULK", 5: "HKK",
        6: "JHM", 7: "MSK", 14: "OLK", 15: "ZLK", 16: "VYS", 
        17: "PAK", 18: "LBK", 19: "KVK"
    }
    df_cleaned['region'] = df_cleaned['p4a'].map(region_map)

    # Odstranění duplikátů na základě sloupce `p1`
    df_cleaned = df_cleaned.drop_duplicates(subset='p1')

    # Výpočet velikosti datového rámce po úpravě (verbose mód)
    if verbose:
        total_size_bytes = df_cleaned.memory_usage(deep=True).sum()
        total_size_mb = total_size_bytes / 10**6  # Převod na MB
        print(f"new_size={total_size_mb:.1f} MB")

    return df_cleaned

# Ukol 3: počty nehod v jednotlivých regionech podle stavu vozovky
def plot_state(df: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    """
    Vytvoří graf počtu nehod v jednotlivých regionech podle stavu vozovky.

    Tato funkce zobrazuje barové grafy pro jednotlivé stavy vozovky v různých regionech,
    a to v rámci čtyř podgrafů pro různé kategorie stavu vozovky.

    Args:
        df (pd.DataFrame): DataFrame po zpracování pomocí funkce `parse_data()`, obsahující data o nehodách.
        fig_location (str): Cesta k uložení obrázku grafu. Pokud None, graf nebude uložen.
        show_figure (bool): Pokud True, zobrazí graf na obrazovce. Výchozí hodnota je False.

    Returns:
        None: Funkce pouze vykreslí a uloží graf.
    """
    # Mapování stavů vozovky
    state_map = {
        1: "Povrch suchý", 2: "Povrch suchý",
        3: "Povrch mokrý", 4: "Na vozovce je bláto",
        5: "Na vozovce je náledí, ujetý sníh", 6: "Na vozovce je náledí, ujetý sníh"
    }
    
    # Přidání popisu stavu vozovky
    df['road_state'] = df['p16'].map(state_map)
    
    # Filtrace platných hodnot (1 až 6) ve sloupci p16
    df_filtered = df[df['road_state'].notna()]
    
    # Agregace: počty nehod podle regionu a stavu vozovky
    data_agg = df_filtered.groupby(['region', 'road_state']).size().reset_index(name='count')
    
    # Vytvoření 4 podgrafů (pro každou kategorii stavů vozovky)
    road_states = [
        "Povrch suchý", "Povrch mokrý",
        "Na vozovce je bláto", "Na vozovce je náledí, ujetý sníh"
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes = axes.flatten()

    sns.set_style('whitegrid')
    plt.rc('font', size=12)
    
    for i, state in enumerate(road_states):
        ax = axes[i]
        # Filtrace dat pro konkrétní stav vozovky
        state_data = data_agg[data_agg['road_state'] == state]
        
        # Vykreslení barového grafu
        sns.barplot(
            data=state_data, x='region', y='count', ax=ax, 
            palette='bright', hue='region', dodge=False
        )

        # Nastavení titulku a popisků
        ax.set_title(f"Stav vozovky: {state}", fontsize=13, fontweight='bold')
        ax.set_xlabel("Region", fontsize=12)
        ax.set_ylabel("Počet nehod", fontsize=12)

        # Úprava otočení popisků osy X
        ax.tick_params(axis='x', rotation=45)

        # Nastavení pozadí podgrafu
        ax.set_facecolor("#f0f0f0")  # Nastavení pozadí podgrafu

    # Nastavení hlavního titulku
    fig.suptitle("Počty nehod podle stavu vozovky a regionů", fontsize=16)
    
    # Uložení grafu, pokud je zadaný fig_location
    if fig_location:
        plt.savefig(fig_location, bbox_inches='tight')
    
    # Zobrazení grafu, pokud je show_figure True
    if show_figure:
        plt.show()
    
    # Zavření grafu po vykreslení
    plt.close(fig)

# Ukol4: alkohol a následky v krajích
def plot_alcohol(df: pd.DataFrame, df_consequences: pd.DataFrame, 
                 fig_location: str = None, show_figure: bool = False):
    """
    Vytvoří grafy počtu nehod pod vlivem alkoholu v jednotlivých regionech a jejich následky.

    Funkce vykreslí grafy počtu nehod pod vlivem alkoholu pro různé úrovně zranění (od bez zranění po usmrcení),
    rozdělené podle role účastníka (řidič/spolujezdec) a regionu. Pro každou úroveň zranění je vytvořen samostatný graf.

    Args:
        df (pd.DataFrame): DataFrame obsahující informace o nehodách.
        df_consequences (pd.DataFrame): DataFrame obsahující informace o následcích nehod.
        fig_location (str): Cesta k uložení obrázku grafu. Pokud None, graf nebude uložen.
        show_figure (bool): Pokud True, zobrazí graf na obrazovce. Výchozí hodnota je False.

    Returns:
        None: Funkce pouze vykreslí a uloží graf.
    """
    # Spojení dat na základě sloupce p1
    merged_df = df.merge(df_consequences, on="p1", how="inner")
    
    # Filtrace pouze nehod pod vlivem alkoholu a s platným časem
    filtered_df = merged_df.loc[(merged_df['p11'] >= 3) & (merged_df['p2a'].notna())].copy()
    
    # Přidání sloupce pro roli účastníka (řidič nebo spolujezdec)
    filtered_df['role'] = filtered_df['p59a'].apply(lambda x: 'Řidič' if x == 1 else 'Spolujezdec')
    
    # Přidání úrovně zranění do samostatného sloupce
    injury_levels = {
        1: "Usmrcení", 2: "Těžké zranění", 3: "Lehké zranění", 4: "Bez zranění"
    }
    filtered_df['injury_level'] = filtered_df['p59g'].map(injury_levels)
    
    # Odstranění záznamů s neplatnými úrovněmi zranění
    filtered_df = filtered_df[filtered_df['injury_level'].notna()]
    
    # Agregace dat podle regionu, úrovně zranění a role
    aggregated_data = (
        filtered_df.groupby(['region', 'injury_level', 'role'])
        .size()
        .reset_index(name='count')
    )
    
    # Nastavení pořadí úrovní zranění
    injury_order = ["Bez zranění", "Lehké zranění", "Těžké zranění", "Usmrcení"]
    aggregated_data['injury_level'] = pd.Categorical(
        aggregated_data['injury_level'], categories=injury_order, ordered=True
    )
    
    # Vykreslení grafů
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    fig.suptitle("Počet nehod pod vlivem alkoholu v regionech", fontsize=16, fontweight='bold', x=0.44)
    fig.subplots_adjust(right=0.8)  # Rezerva pro legendu
    
    # Mřížka
    sns.set_style({"grid.color": ".1"})
    
    for i, injury_level in enumerate(injury_order):
        ax = axes[i // 2, i % 2]
        subset = aggregated_data[aggregated_data['injury_level'] == injury_level]
        
        sns.barplot(
            data=subset,
            x="region", y="count", hue="role", ax=ax,
            palette="muted"
        )
        
        # Nastavení popisků a titulku
        ax.set_title(f"Následky nehody: {injury_level}", fontsize=14)
        ax.set_xlabel("Region", fontsize=12)
        ax.set_ylabel("Počet zranění", fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        
        # Skryjeme legendy u jednotlivých podgrafů
        ax.legend_.remove()
        
        # Přidání černého ohraničení kolem každého podgrafu
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
    
    # Přidání jedné společné legendy bez ohraničení
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, title="Role účastníka",
        loc='center right', bbox_to_anchor=(0.9, 0.5), frameon=False
    )
    
    # Uložení a zobrazení grafu
    plt.tight_layout(rect=[0, 0, 0.8, 0.95])  # Rezerva pro nadpis a legendu
    if fig_location:
        plt.savefig(fig_location, bbox_inches='tight')
    if show_figure:
        plt.show()
    
    # Zavření grafu
    plt.close(fig)

# Ukol 5: Druh nehody (srážky) v čase
def plot_type(df: pd.DataFrame, fig_location: str = None,
              show_figure: bool = False):
    """
    Vytvoří grafy počtu nehod podle druhu srážky v jednotlivých regionech v čase.

    Funkce vykreslí časovou analýzu počtu nehod podle druhu srážky (např. s vozidlem, s chodcem) pro vybrané kraje
    (Praha, Středočeský, Jihomoravský a Moravskoslezský) v období od ledna 2023 do září 2024.

    Args:
        df (pd.DataFrame): DataFrame obsahující informace o nehodách.
        fig_location (str): Cesta k uložení obrázku grafu. Pokud None, graf nebude uložen.
        show_figure (bool): Pokud True, zobrazí graf na obrazovce. Výchozí hodnota je False.

    Returns:
        None: Funkce pouze vykreslí a uloží graf.
    """
    # Výběr čtyř krajů
    selected_regions = ['PHA', 'STC', 'JHM', 'MSK']  # Například Praha, Středočeský, Jihomoravský, Moravskoslezský
    
    # Filtrace pouze vybraných krajů a srážek (p6 > 0)
    filtered_df = df[(df['region'].isin(selected_regions)) & (df['p6'] > 0)].copy()
    
    # Přidání textových hodnot pro druh srážky
    collision_types = {
        1: "s nekolejovým vozidlem", 2: "s vozidlem zaparkovaným", 3: "s pevnou překážkou", 4: "s chodcem",
        5: "s lesní zvěří", 6: "s domácím zvířetem", 7: "s vlakem", 8: "s tramvají"
    }
    filtered_df['collision_type'] = filtered_df['p6'].map(collision_types)
    
    # Převod data na měsíční úroveň
    filtered_df['month'] = pd.to_datetime(filtered_df['date']).dt.to_period('M')
    
    # Agregace počtu nehod podle měsíce, regionu a typu srážky
    monthly_data = (
        filtered_df.groupby(['region', 'month', 'collision_type'])
        .size()
        .reset_index(name='count')
    )
    
    # Zajistíme, že 'count' je číselný
    monthly_data['count'] = pd.to_numeric(monthly_data['count'])

    # Přeformátování tabulky na stacked formát
    monthly_pivot = monthly_data.pivot_table(
        index=['month', 'region'],
        columns='collision_type',
        values='count',
        fill_value=0
    ).stack().reset_index(name='count')

    # Filtrace dat na období od 1.1.2023 do 1.10.2024
    start_date = '2023-01'
    end_date = '2024-09'
    filtered_time = monthly_pivot.loc[
        (monthly_pivot['month'] >= start_date) & (monthly_pivot['month'] <= end_date)
    ].copy()

    # Převod na správný formát
    filtered_time['month'] = filtered_time['month'].dt.to_timestamp()
    
    # Nastavení vykreslování
    sns.set_style("white")
    g = sns.relplot(
        data=filtered_time,
        x='month', y='count', hue='collision_type', kind='line',
        col='region', col_wrap=2, height=5, aspect=1.5, palette='tab10'
    )
    
    # Nastavení os a popisků pro každý podgraf
    for ax in g.axes.flat:
        # Nastavení formátu dat na ose X jako MM/YYYY
        ax.xaxis.set_major_formatter(dates.DateFormatter('%m/%Y'))
        
        # Otočení hodnot osy X o 45 stupňů
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
        
        # Nastavení názvů os
        ax.set_xlabel("Datum nehod (MM/YYYY)", fontsize=10)
        ax.set_ylabel("Počet nehod", fontsize=10)

        # Přidání orámování podgrafu ze všech stran
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('black')

        # Vypnutí mřížky
        ax.grid(False)
    
    # Přidání hlavního nadpisu
    g.fig.suptitle("Druhy nehod v letech (2023–2024)", fontsize=16, fontweight='bold')
    g.fig.subplots_adjust(top=0.9)  # Rezerva pro nadpis
    
    # Úprava legendy
    g._legend.set_bbox_to_anchor((1, 0.5))  # Posunutí legendy vpravo mimo graf
    g._legend.set_title("Druh nehody")  # Titulek legendy
    g._legend.set_frame_on(False)  # Odebrání rámu legendy
    
    # Uložení a zobrazení grafu
    if fig_location:
        plt.savefig(fig_location, bbox_inches='tight')
    if show_figure:
        plt.show()
    
    # Zavření grafu
    plt.close(g.fig)

if __name__ == "__main__":
    df = load_data("part02/data_23_24.zip", "nehody")
    df_consequences = load_data("part02/data_23_24.zip", "nasledky")
    df2 = parse_data(df, True)

    plot_state(df2, "part02/figures/01_state.png", show_figure=False)
    plot_alcohol(df2, df_consequences, "part02/figures/02_alcohol.png", show_figure=False)
    plot_type(df2, "part02/figures/03_type.png")
