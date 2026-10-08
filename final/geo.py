#!/usr/bin/python3.10
# coding=utf-8
# %%%
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np
import matplotlib
from shapely.geometry import Point, Polygon
from sklearn.cluster import KMeans


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


def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None,
             show_figure: bool = False):
    """
    Vykreslí graf s pozicemi nehod v Jihomoravském kraji, které byly zaviněny pod vlivem alkoholu, 
    pro dva vybrané měsíce (květen a říjen).

    Funkce vizualizuje nehody v Jihomoravském kraji (kraj 'JHM'), kde byly nehody způsobeny pod 
    vlivem alkoholu (sloupec 'p11' má hodnotu větší nebo rovno 4). Vytvoří dva grafy pro měsíc 
    květen a říjen, kde jsou zobrazeny jednotlivé nehody na mapě s podkladovou mapou OpenStreetMap 
    v systému souřadnic EPSG:3857.

    Parametry:
    -----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame obsahující informace o nehodách včetně sloupce 'region' (Jihomoravský kraj) 
        a 'p11' (vztahující se k alkoholu).

    fig_location : str, optional
        Cesta k souboru pro uložení grafu. Pokud není zadána, graf nebude uložen. Výchozí hodnota je None.

    show_figure : bool, optional
        Pokud je True, graf bude zobrazen. Pokud je False, graf nebude zobrazen. Výchozí hodnota je False.

    Returns:
    --------
    None
        Funkce nevrací žádnou hodnotu. Vykreslí a/nebo uloží graf na základě specifikovaných parametrů.

    Notes:
    ------
    - Funkce filtruje data pro Jihomoravský kraj (kraj 'JHM') a nehody, které byly zaviněny pod vlivem alkoholu (p11 >= 4).
    - Data jsou rozdělena do dvou měsíců: květen (5) a říjen (10).
    - Mapa zobrazuje souřadnice nehod s podkladovou mapou OpenStreetMap v systému souřadnic EPSG:3857.
    - Grafy pro každý měsíc jsou zobrazeny vedle sebe (s dvěma osami pro každý měsíc).
    """

    # Převod sloupce date na datetime a extrakce měsíce
    gdf['month'] = pd.to_datetime(gdf['date']).dt.month

    # Filtrace pro Jihomoravský kraj (kraj 'JHM') a nehody pod vlivem alkoholu (p11 >= 4)
    gdf_jhm = gdf[(gdf['region'] == 'JHM') & (gdf['p11'] >= 4)]
    
    # Vybereme dva měsíce (např. květen a říjen)
    gdf_may = gdf_jhm[gdf_jhm['month'] == 5]
    gdf_october = gdf_jhm[gdf_jhm['month'] == 10]
    
    # Transformace na CRS EPSG:3857 pro kompatibilitu s podkladovou mapou
    gdf_may = gdf_may.to_crs(epsg=3857)
    gdf_october = gdf_october.to_crs(epsg=3857)
    
    # Kombinované limity pro stejné měřítko grafů
    combined_bounds = [
        min(gdf_may.total_bounds[0], gdf_october.total_bounds[0]),
        min(gdf_may.total_bounds[1], gdf_october.total_bounds[1]),
        max(gdf_may.total_bounds[2], gdf_october.total_bounds[2]),
        max(gdf_may.total_bounds[3], gdf_october.total_bounds[3]),
    ]
    
    # Přidání okrajů k hranicím
    margin_factor = 0.05  # Přidáme 5 % prostoru kolem hranic
    x_range = combined_bounds[2] - combined_bounds[0]
    y_range = combined_bounds[3] - combined_bounds[1]
    expanded_bounds = [
        combined_bounds[0] - margin_factor * x_range,
        combined_bounds[1] - margin_factor * y_range,
        combined_bounds[2] + margin_factor * x_range,
        combined_bounds[3] + margin_factor * y_range,
    ]
    
    # Nastavení grafu
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    # Květen (May)
    ax = axes[0]
    gdf_may.plot(ax=ax, color='red', markersize=4)
    ctx.add_basemap(ax, crs=gdf_may.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
    ax.set_title("JHM kraj - Nehody pod vlivem alkoholu (Květen)")
    ax.set_xlim(expanded_bounds[0], expanded_bounds[2])
    ax.set_ylim(expanded_bounds[1], expanded_bounds[3])
    ax.set_axis_off()  # Skrytí os
    
    # Říjen (October)
    ax = axes[1]
    gdf_october.plot(ax=ax, color='red', markersize=4)
    ctx.add_basemap(ax, crs=gdf_october.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
    ax.set_title("JHM kraj - Nehody pod vlivem alkoholu (Říjen)")
    ax.set_xlim(expanded_bounds[0], expanded_bounds[2])
    ax.set_ylim(expanded_bounds[1], expanded_bounds[3])
    ax.set_axis_off()  # Skrytí os
    
    # Uložit do souboru, pokud je specifikován
    if fig_location:
        plt.savefig(fig_location, bbox_inches='tight')
    
    # Zobrazit graf, pokud je show_figure True
    if show_figure:
        plt.show()
    
    # Zavřít graf
    plt.close(fig)


def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """
    Vykreslí mapu s clustery nehod v Jihomoravském kraji, zaměřenou na nehody zaviněné zvěří, 
    využívající shlukování KMeans a vytvoření oblastí kolem těchto clusterů pomocí convex hull.

    Funkce filtrace dat pro Jihomoravský kraj (region 'JHM') a nehody zaviněné zvěří (p10 == 4) 
    a následně provádí shlukování těchto nehod pomocí algoritmu KMeans do zadaného počtu clusterů. 
    Každý cluster je poté obklopen oblastí vytvořenou pomocí convex hull, která zobrazuje území, 
    kde došlo k větší koncentraci nehod. Na mapě jsou zobrazeny body nehod a příslušné oblasti 
    kolem každého clusteru.

    Parametry:
    -----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame obsahující informace o nehodách, včetně sloupce 'region' (Jihomoravský kraj) 
        a 'p10' (nehody zaviněné zvěří).

    fig_location : str, optional
        Cesta k souboru pro uložení grafu. Pokud není zadána, graf nebude uložen. Výchozí hodnota je None.

    show_figure : bool, optional
        Pokud je True, graf bude zobrazen. Pokud je False, graf nebude zobrazen. Výchozí hodnota je False.

    Returns:
    --------
    None
        Funkce nevrací žádnou hodnotu. Vykreslí a/nebo uloží graf na základě specifikovaných parametrů.

    Notes:
    ------
    - Funkce filtruje data pro Jihomoravský kraj (region 'JHM') a nehody zaviněné zvěří (p10 == 4).
    - KMeans je použit pro shlukování bodů do 'n_clusters' clusterů.
    - Pro každý cluster je vytvořena oblast pomocí convex hull.
    - Na mapě jsou zobrazeny nehody (červené body) a oblasti (poloprůhledné) kolem clusterů.
    - Používá podkladovou mapu OpenStreetMap v systému souřadnic EPSG:3857.
    """

    # Filtrace dat pro Jihomoravský kraj (JHM) a nehody zaviněné zvěří (p10 == 4)
    gdf_jhm = gdf[(gdf['region'] == 'JHM') & (gdf['p10'] == 4)]
    
    # Ujistíme se, že souřadnice nejsou NaN nebo Inf
    gdf_jhm = gdf_jhm[gdf_jhm.geometry.is_valid]
    
    # Transformace do souřadnicového systému EPSG:3857 (pro kompatibilitu s mapou)
    gdf_jhm = gdf_jhm.to_crs(epsg=3857)
    
    # Extrakce souřadnic pro shlukování
    coords = np.array([point.coords[0] for point in gdf_jhm.geometry])
    
    # Použití KMeans pro shlukování do 'n_clusters' clusterů
    n_clusters = 8
    kmeans = KMeans(n_clusters, random_state=0)
    gdf_jhm['cluster'] = kmeans.fit_predict(coords)
    
    # Počet nehod v jednotlivých clusterech
    cluster_sizes = gdf_jhm.groupby('cluster').size()
    
    # Vytvoření oblasti kolem každého clusteru pomocí convex hull
    hulls = []
    cmap = matplotlib.colormaps['viridis']  # Používáme viridis pro plynulý přechod barev
    norm = plt.Normalize(vmin=cluster_sizes.min(), vmax=cluster_sizes.max())  # Normalizace pro barevný přechod

    for cluster_id in range(n_clusters):
        # Všechny souřadnice bodů v daném clusteru
        cluster_points = coords[gdf_jhm['cluster'] == cluster_id]
        
        # Vytvoření convex hull pro body v daném clusteru
        if len(cluster_points) >= 3:
            polygon = Polygon(cluster_points)
            hulls.append(polygon.convex_hull)
    
    # Vykreslení
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Zobrazení oblastí kolem clusterů
    for i, (hull, cluster_id) in enumerate(zip(hulls, range(n_clusters))):
        if hull.is_valid:
            # Barva podle počtu nehod v daném clusteru
            color = cmap(norm(cluster_sizes[cluster_id]))
            ax.fill(*hull.exterior.xy, alpha=0.3, color=color, label=f'Cluster {i+1}')
    
    # Vykreslení bodů (nehody) ve všech clusterech s červenou barvou
    ax.scatter(coords[:, 0], coords[:, 1], color='red', s=10, label='Nehody', zorder=5)
    
    # Přidání podkladové mapy
    ctx.add_basemap(ax, crs=gdf_jhm.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
    
    # Nastavení titulků, legendy a odebrání os
    ax.set_title("Nehody v JHM kraji zaviněný lesní zvěří")
    ax.set_axis_off()
    
    # Přidání barevné čáry (ColorBar) s roztahováním
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Potřebné pro vytvoření colorbaru
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.04, pad=0.01)  # Zvětšeno fraction
    cbar.set_label('Počet nehod v úseku')
    
    # Uložit do souboru, pokud je specifikován
    if fig_location:
        plt.savefig(fig_location, bbox_inches="tight")
    
    # Zobrazit graf, pokud je show_figure True
    if show_figure:
        plt.show()
    
    # Zavřít graf
    plt.close(fig)

if __name__ == "__main__":
    # zde muzete delat libovolne modifikace
    df_accidents = pd.read_pickle("final/accidents.pkl.gz")
    df_locations = pd.read_pickle("final/locations.pkl.gz")

    gdf = make_geo(df_accidents, df_locations)

    plot_geo(gdf, "final/figures/geo1.png", False)
    plot_cluster(gdf, "final/figures/geo2.png", False)

    # testovani splneni zadani
    import os
    assert os.path.exists("final/figures/geo1.png")
    assert os.path.exists("final/figures/geo2.png")
