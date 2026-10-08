import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point


def make_geo(df_accidents: pd.DataFrame, df_locations: pd.DataFrame) -> geopandas.GeoDataFrame:
    """
    Vytvoří GeoDataFrame obsahující geolokační informace o nehodách spojené s daty o lokalitách.

    Funkce bere dva pandas DataFrame: jeden obsahující informace o nehodách a druhý obsahující 
    geolokační data (souřadnice). Po spojení těchto dat vytvoří GeoDataFrame s geometrií bodů, 
    která reprezentuje pozice nehod. Souřadnice jsou transformovány do geografického systému souřadnic 
    EPSG:5514 (pro Českou republiku) a jsou zajištěna pouze platná data, která jsou uvnitř území ČR.

    Parameters:
    -----------
    df_accidents : pandas.DataFrame
        DataFrame obsahující informace o nehodách, včetně sloupce 'p1', který obsahuje ID lokalit.
    
    df_locations : pandas.DataFrame
        DataFrame obsahující informace o geografických souřadnicích lokalit (sloupce 'd' a 'e' pro souřadnice).

    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame obsahující data o nehodách spojené s geolokačními informacemi, 
        transformovanými do CRS EPSG:5514 a filtrováno na území ČR.

    Notes:
    ------
    - Funkce předpokládá, že souřadnice lokalit jsou ve formátu, kde 'd' představuje zeměpisnou šířku 
      a 'e' zeměpisnou délku. Pokud je 'd' menší než 'e', souřadnice jsou prohozeny.
    - Data jsou filtrována tak, aby obsahovala pouze body uvnitř území České republiky.
    """
    
    # Odstranění neplatných souřadnic
    df_locations = df_locations[(df_locations['d'] != 0) & (df_locations['e'] != 0)]
    
    # Geometrie s prohozením souřadnic
    geometry = [
        Point(d, e) if d >= e else Point(e, d)
        for d, e in zip(df_locations['d'], df_locations['e'])
    ]
    
    # Vytvoření GeoDataFrame
    gdf_locations = geopandas.GeoDataFrame(df_locations.copy(), geometry=geometry)
    
    # Nastavení CRS (EPSG:5514 pro ČR)
    gdf_locations.set_crs(epsg=5514, inplace=True)
    
    # Spojení s df_accidents podle klíče 'p1'
    gdf = pd.merge(df_accidents, gdf_locations, how='inner', on='p1')

    # Reset indexu po spojení (odstranění duplicit v indexu)
    gdf.reset_index(drop=True, inplace=True)

    # Převod na GeoDataFrame (znovu přiřazení geometrie)
    gdf = geopandas.GeoDataFrame(gdf, geometry=gdf['geometry'], crs=gdf_locations.crs)
 
    # Kontrola, že všechny body jsou v ČR (buffer okolo bodu v ČR)
    cz_center = Point(15, 50)  # Střed ČR v EPSG:4326
    cz_area = geopandas.GeoSeries([cz_center], crs="EPSG:4326").to_crs(epsg=5514).buffer(1e6)[0]  # Buffer 1 milion metrů
    gdf = gdf[gdf.geometry.within(cz_area)]
    
    return gdf


def plot_geo_winter_weather(gdf: geopandas.GeoDataFrame, fig_location: str = None,
                            show_figure: bool = False):
    """
    Vykreslí mapu s pozicemi nehod v kraji Vysočina (VYS) podle povětrnostních podmínek (p18)
    během zimních měsíců (prosinec, leden, únor).

    Funkce vizualizuje nehody v kraji Vysočina (region 'VYS') a rozlišuje je podle
    povětrnostních podmínek v době nehody (sloupec 'p18'). Zobrazuje pouze nehody během
    zimních měsíců.

    Parametry:
    -----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame obsahující informace o nehodách včetně sloupce 'region' (Vysočina),
        'p18' (povětrnostní podmínky) a 'date' (datum nehody).

    fig_location : str, optional
        Cesta k souboru pro uložení grafu. Pokud není zadána, graf nebude uložen. Výchozí hodnota je None.

    show_figure : bool, optional
        Pokud je True, graf bude zobrazen. Pokud je False, graf nebude zobrazen. Výchozí hodnota je False.

    Returns:
    --------
    None
        Funkce nevrací žádnou hodnotu. Vykreslí a/nebo uloží graf na základě specifikovaných parametrů.
    """
    # Převod sloupce 'date' na datetime a extrakce měsíce
    gdf['month'] = pd.to_datetime(gdf['date']).dt.month
    
    # Filtrace pro kraj Vysočina (VYS) a zimní měsíce
    winter_months = [12, 1, 2]
    gdf_vys = gdf[(gdf['region'] == 'VYS') & (gdf['month'].isin(winter_months))]
    
    # Transformace na CRS EPSG:3857 pro kompatibilitu s podkladovou mapou
    gdf_vys = gdf_vys.to_crs(epsg=3857)
    
    # Nastavení barev pro různé hodnoty p18
    weather_conditions = {
        1: "Neztížené",
        2: "Mlha",
        3: "Mrholení",
        4: "Déšť",
        5: "Sněžení",
        6: "Náledí",
        7: "Nárazový vítr",
        0: "Jiné"
    }
    # Explicitní pořadí hodnot pro legendu
    ordered_conditions = [1, 2, 3, 4, 5, 6, 7, 0]
    colors = plt.cm.tab10(range(len(ordered_conditions)))  # Použijeme paletu tab10
    color_map = {condition: color for condition, color in zip(ordered_conditions, colors)}
    
    # Přidání barvy podle povětrnostních podmínek
    gdf_vys['color'] = gdf_vys['p18'].map(color_map)
    
    # Nastavení grafu
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Vykreslení nehod podle pořadí
    for condition in ordered_conditions:
        subset = gdf_vys[gdf_vys['p18'] == condition]
        label = weather_conditions.get(condition, "Neznámé")  # Pouze popis bez čísla
        if not subset.empty:  # Pouze pokud existují data pro danou podmínku
            subset.plot(ax=ax, color=color_map[condition], markersize=4, label=label)
    
    # Přidání podkladové mapy
    ctx.add_basemap(ax, crs=gdf_vys.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
    
    # Nastavení hranic grafu
    ax.set_xlim(gdf_vys.total_bounds[0], gdf_vys.total_bounds[2])
    ax.set_ylim(gdf_vys.total_bounds[1], gdf_vys.total_bounds[3])
    
    # Přidání titulu, legendy a vypnutí os
    ax.set_title("Kraj Vysočina - Nehody podle povětrnostních podmínek (zimní měsíce)")
    ax.legend(loc='lower left', title="Povětrnostní podmínky", markerscale=2)
    ax.set_axis_off()
    
    # Uložit do souboru, pokud je specifikován
    if fig_location:
        plt.savefig(fig_location, bbox_inches='tight')
    
    # Zobrazit graf, pokud je show_figure True
    if show_figure:
        plt.show()
    
    # Zavřít graf
    plt.close(fig)


def generate_weather_table_all_seasons_with_average(gdf: geopandas.GeoDataFrame, output_format: str = "text", file_path: str = None):
    """
    Generuje tabulky s povětrnostními podmínkami a jejich statistikami (počet, procentuální zastoupení)
    pro všechna roční období (zima, jaro, léto, podzim) a vypočítá průměrné procentuální zastoupení
    pro specifické kategorie (Náledí, Sněžení, Neztížené) mimo zimní měsíce.

    Parametry:
    -----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame obsahující informace o nehodách včetně sloupců 'p18' (povětrnostní podmínky) a 'month' (měsíc nehody).

    output_format : str, optional
        Formát výstupu tabulky: "text" (default), "csv", "latex".

    file_path : str, optional
        Cesta k souboru, pokud je výstup uložen (použije se pouze u 'csv' a 'latex').

    Returns:
    --------
    None
        Tabulky jsou vytištěny do konzole nebo uloženy do souborů podle zvoleného formátu.
    """
    # Mapování povětrnostních podmínek
    weather_conditions = {
        1: "Neztížené",
        2: "Mlha",
        3: "Mrholení",
        4: "Déšť",
        5: "Sněžení",
        6: "Náledí",
        7: "Nárazový vítr",
        0: "Jiné"
    }
    
    # Definice ročních období
    seasons = {
        "Zima": [12, 1, 2],
        "Jaro": [3, 4, 5],
        "Léto": [6, 7, 8],
        "Podzim": [9, 10, 11]
    }
    
    # Seznamy pro uchování statistik pro konkrétní kategorie
    target_conditions = [1, 5, 6]  # Neztížené, Sněžení, Náledí
    season_stats = {}
    average_stats = {condition: [] for condition in target_conditions}

    # Pro každé období vypočítáme statistiky
    for season_name, months in seasons.items():
        gdf_season = gdf[gdf['month'].isin(months)]
        
        # Výpočet statistik
        condition_counts = gdf_season['p18'].value_counts().sort_index()
        total_count = condition_counts.sum()
        condition_stats = pd.DataFrame({
            "Povětrnostní podmínky": [weather_conditions.get(cond, "Neznámé") for cond in condition_counts.index],
            "Množství": condition_counts.values,
            "Procentuální zastoupení (%)": (condition_counts.values / total_count * 100).round(2)
        })
        condition_stats.loc[len(condition_stats)] = ["Celkový součet", total_count, 100.0]
        season_stats[season_name] = condition_stats
        
        # Výpočet průměrů pro specifické kategorie (mimo Zima)
        if season_name != "Zima":
            for condition in target_conditions:
                condition_value = condition_stats[condition_stats["Povětrnostní podmínky"] == weather_conditions[condition]]
                if not condition_value.empty:
                    percentage = condition_value["Procentuální zastoupení (%)"].values[0]
                    average_stats[condition].append(percentage)

        # Výstup dle požadovaného formátu
        if output_format == "text":
            print(f"\n{season_name}:\n")
            print(condition_stats.to_string(index=False, justify="center", col_space=15))
        elif output_format == "csv" and file_path:
            season_file_path = file_path.replace(".csv", f"_{season_name}.csv")
            condition_stats.to_csv(season_file_path, index=False, sep=";")
            print(f"Tabulka pro {season_name} byla uložena do souboru: {season_file_path}")
        elif output_format == "latex" and file_path:
            season_file_path = file_path.replace(".tex", f"_{season_name}.tex")
            with open(season_file_path, "w") as f:
                f.write(condition_stats.to_latex(index=False, caption=f"Statistiky povětrnostních podmínek pro {season_name}"))
            print(f"Tabulka pro {season_name} byla uložena do souboru: {season_file_path}")

    # Vypočítání průměrných hodnot pro kategorie mimo zimu
    print("\nPrůměrné procentuální zastoupení pro Neztížené, Sněžení a Náledí (mimo zimu):")
    for condition in target_conditions:
        if average_stats[condition]:
            avg_percentage = sum(average_stats[condition]) / len(average_stats[condition])
            print(f"{weather_conditions[condition]}: {avg_percentage:.2f}%")
        else:
            print(f"{weather_conditions[condition]}: Není dostupné.")


if __name__ == "__main__":
    df_accidents = pd.read_pickle("final/accidents.pkl.gz")
    df_locations = pd.read_pickle("final/locations.pkl.gz")

    gdf = make_geo(df_accidents, df_locations)

    plot_geo_winter_weather(gdf, "final/figures/fig.png", False)
    generate_weather_table_all_seasons_with_average(gdf, output_format="text")
