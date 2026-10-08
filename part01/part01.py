#!/usr/bin/env python3
"""
IZV cast1 projektu
Autor: Michael Kubát
"""

import requests
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.collections import LineCollection
from typing import List, Dict, Any


def distance(a: np.array, b: np.array) -> np.array:
    """
    Vypocte eukleidovskou vzdalenost mezi odpovidajicimi body ve dvou polich.

    Tato funkce spocita eukleidovskou vzdalenost pro kazdou dvojici odpovidajicich
    bodu v polich 'a' a 'b'. Obe 'a' a 'b' by mely byt dvourozmerna pole, kde
    radky reprezentuji body a sloupce reprezentuji souradnice (napr. x, y, z).

    Parametry
    ---------
    a : np.array
        Dvourozmerne numpy pole tvaru (n, m), kde 'n' je pocet bodu a
        'm' je pocet rozmeru pro kazdy bod.
    b : np.array
        Dvourozmerne numpy pole tvaru (n, m), kde 'n' je pocet bodu a
        'm' je pocet rozmeru pro kazdy bod.

    Navratova hodnota
    -----------------
    np.array
        Jednorozmerne numpy pole tvaru (n,), kde kazdy prvek reprezentuje eukleidovskou
        vzdalenost mezi odpovidajicimi body v 'a' a 'b'.
        
    Vyjimky
    -------
    ValueError
        Pokud 'a' a 'b' maji ruzne tvary.

    Priklady
    --------
    >>> import numpy as np
    >>> a = np.array([[0, 0], [1, 1], [2, 2]])
    >>> b = np.array([[0, 1], [1, 2], [2, 3]])
    >>> distance(a, b)
    array([1. , 1. , 1.41421356])

    """
    # Rozdíl mezi odpovídajícími body
    diff = a - b
    # Součet čtverců rozdílů pro každou dvojici bodů a odmocnění výsledku
    distances = np.sqrt(np.sum(diff**2, axis=1))
    return distances


def generate_graph(a: List[float], show_figure: bool = False, save_path: str | None = None):
    """
    Vygeneruje graf funkce f_a(x) pro ruzne hodnoty parametru 'a' a zobrazi jej nebo ulozi jako obrazek.

    Funkce vykresli graf pro kazdou hodnotu v seznamu 'a', kde 'f_a(x) = a^2 * sin(x)' 
    v intervalu [0, 6π]. Kazda křivka je vykreslena s odpovidajici barvou a vyplni
    plochu pod krivkou az k ose x. Graf muze byt zobrazen primo, ulozen jako soubor, 
    nebo oboji.

    Parametry
    ---------
    a : List[float]
        Seznam hodnot parametru 'a', pro ktere se generuje graf.
    show_figure : bool, volitelne
        Pokud je True, zobrazi graf po jeho vytvoreni. Vychozi hodnota je False.
    save_path : str nebo None, volitelne
        Cesta k souboru, kam se graf ulozi. Pokud je None, graf se neulozi.

    Navratova hodnota
    -----------------
    None
        Tato funkce nic nevraci. 

    Priklady
    --------
    >>> generate_graph([3, 4, 7], show_figure=True, save_path='graf.png')

    """
    # Definice x v intervalu [0, 6π]
    x = np.linspace(0, 6 * np.pi, 500)

    # Výpočet hodnot f_a(x) pro každé 'a' pomocí broadcastingu
    a = np.array(a)[:, np.newaxis]  # Přidáme novou osu pro správné broadcastování
    y = a**2 * np.sin(x)

    # Barvy pro jednotlivé hodnoty 'a'
    line_color_map = {7: "tab:blue", 4: "tab:orange", 3: "tab:green"}
    fill_color_map = {7: "lightsteelblue", 4: "wheat", 3: "darkseagreen"}

    # Vykreslení grafu
    plt.figure(figsize=(10, 6))
    for i in range(len(a)):
        a_value = a[i, 0].item()  # Získání skalarové hodnoty 'a'
        line_color = line_color_map.get(a_value, "black") 
        fill_color = fill_color_map.get(a_value, "lightgray")

        plt.plot(x, y[i], color=line_color, label=fr"$\gamma _{{{a_value}}}(x)$", linewidth=2)
        
        # Výplň od osy x po křivku bez překrytí
        plt.fill_between(x, 0, y[i], color=fill_color, alpha=0.3)

    # Nastavení osy x a popisků
    plt.xlabel("x", fontsize=12)
    plt.ylabel(r"$f_a(x)$", fontsize=12)
    plt.xlim(0, 6 * np.pi)

    # Rozsah osy y
    max_y = max(y.max(axis=0)) 
    min_y = -max_y
    plt.ylim(min_y * 1.1, max_y * 1.1)

    # Nastavení osy x
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(np.pi / 2))

    # Formátovací funkce pro osu x
    def format_func(value, tick_number):
        N = int(np.round(2 * value / np.pi))
        if N == 0:
            return "0"
        elif N == 1:
            return r"$\frac{1}{2}\pi$"
        elif N == 2:
            return r"$\pi$"
        elif N % 2 > 0:
            return fr"$\frac{{{N}}}{{2}}\pi$"
        else:
            return fr"${N // 2}\pi$"

    ax.xaxis.set_major_formatter(FuncFormatter(format_func))

    # Upravená legenda
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=len(a), fontsize=12, frameon=True, 
               edgecolor="lightgray", framealpha=1, facecolor="white")

    # Posun legendy nad graf
    plt.subplots_adjust(top=0.80)

    # Zobrazení nebo uložení grafu
    if save_path:
        plt.savefig(save_path, dpi=300)
    if show_figure:
        plt.show()
    plt.close()


def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    """
    Vygeneruje a vykresli sinusove funkce f1(t), f2(t) a jejich soucet, s volitelnym ulozenim nebo zobrazenim.

    Tato funkce vytvori a vykresli grafy pro dve sinusove funkce, 'f1' a 'f2', v zavislosti na case 't', 
    spolu s grafem jejich souctu 'f_sum'. Soucet je vykreslen s podminenym vybarvenim podle toho, 
    zda prevysuje hodnotu 'f1' nebo podle hodnoty 't'.

    Parametry
    ---------
    show_figure : bool, volitelne
        Pokud je True, graf se zobrazi po jeho vytvoreni. Vychozi hodnota je False.
    save_path : str nebo None, volitelne
        Cesta k souboru, kam se graf ulozi. Pokud je None, graf se neulozi.

    Navratova hodnota
    -----------------
    None
        Tato funkce nic nevraci.

    Priklady
    --------
    >>> generate_sinus(show_figure=True, save_path='sinus_graf.png')

    """
    # Definování intervalu a funkcí
    t = np.arange(0, 100.01, 0.01)  # Rozsah hodnot t
    f1 = 0.5 * np.cos(1 / 50 * np.pi * t)
    f2 = 0.25 * (
        np.sin(np.pi * t)
        + np.sin(3 / 2 * np.pi * t)
    )
    f_sum = f1 + f2

    # Vytvoření hlavního grafu pro přesnější kontrolu pozice podgrafů
    fig = plt.figure(figsize=(10, 8))

    # První podgraf - funkce f1
    ax1 = fig.add_axes([0.1, 0.69, 0.8, 0.2])
    ax1.plot(t, f1, color="tab:blue", linewidth=1.75)
    ax1.set_ylim(-0.8, 0.8)
    ax1.set_yticks(np.arange(-0.8, 1.0, 0.4))
    ax1.set_xlim(0, 100)
    ax1.set_xticks(np.arange(0, 101, 25))
    ax1.tick_params(labelbottom=False)
    ax1.set_ylabel(r"$f_1(t)$", fontsize=10)

    # Druhý podgraf - funkce f2
    ax2 = fig.add_axes([0.1, 0.45, 0.8, 0.2])
    ax2.plot(t, f2, color="tab:blue", linewidth=1.75)
    ax2.set_ylim(-0.8, 0.8)
    ax2.set_yticks(np.arange(-0.8, 1.0, 0.4))
    ax2.set_xlim(0, 100)
    ax2.set_xticks(np.arange(0, 101, 25))
    ax2.tick_params(labelbottom=False)
    ax2.set_ylabel(r"$f_2(t)$", fontsize=10)

    # Třetí podgraf - součet f1 + f2 s podmíněným vybarvením
    ax3 = fig.add_axes([0.1, 0.21, 0.8, 0.2])
    segments = []
    colors = []
    for i in range(len(t) - 1):
        if f_sum[i] > f1[i]:
            color = 'tab:green'
        elif t[i] < 50:
            color = 'tab:red'
        else:
            color = 'tab:orange'
        segments.append([(t[i], f_sum[i]), (t[i + 1], f_sum[i + 1])])
        colors.append(color)

    lc = LineCollection(segments, colors=colors, linewidth=1.75)
    ax3.add_collection(lc)
    ax3.set_ylim(-0.8, 0.8)
    ax3.set_yticks(np.arange(-0.8, 1.0, 0.4))
    ax3.set_xlim(0, 100)
    ax3.set_xticks(np.arange(0, 101, 25))
    ax3.set_ylabel(r"$f_1(t)$ + $f_2(t)$", fontsize=10)

    # Uložení nebo zobrazení grafu
    if save_path:
        plt.savefig(save_path, dpi=300)
    if show_figure:
        plt.show()
    plt.close()


def download_data() -> Dict[str, List[Any]]:
    """
    Stahne a zpracuje data o stanicich z URL a vrati je jako slovnik s pozicemi, 
    zemepisnymi sirkami, delkami a vyskami.

    Funkce stahne HTML stranku obsahujici informace o stanicich, pouzije knihovnu 
    BeautifulSoup pro parsovani HTML, a vybere data o pozicich jednotlivych stanic. 
    Data jsou vracena ve slovniku, kde klice jsou 'positions', 'lats', 'longs', a 
    'heights', pricemz hodnoty jsou seznamy odpovidajici kazdemu klici.

    Navratova hodnota
    -----------------
    Dict[str, List[Any]]
        Slovnik obsahujici seznamy:
        - 'positions' : seznam nazvu mest/stanic.
        - 'lats' : seznam zemepisnych sirek (prazdny). 
        - 'longs' : seznam zemepisnych delek (prazdny).
        - 'heights' : seznam vysky nad morem (prazdny).

    Priklady
    --------
    >>> data = download_data()
    >>> print(data['positions'])  # Vytiskne seznam nazvu mest nebo stanic

    """
    url = "https://ehw.fit.vutbr.cz/izv/st_zemepis_cz.html"
    
    # Stáhnout obsah stránky a nastavit kódování na UTF-8
    response = requests.get(url)
    response.raise_for_status()   # Zkontrolovat, zda bylo stahování úspěšné
    response.encoding = response.apparent_encoding

    # Vytvořit BeautifulSoup objekt pro parsování HTML
    soup = BeautifulSoup(response.text, "html.parser")

    # Inicializovat prázdné seznamy pro jednotlivé hodnoty
    positions = []  # seznam pro města
    lats = []   # seznam pro zeměpisné šířky
    longs = []  # seznam pro zeměpisné délky
    heights = [] # seznam pro výšky

    for row in soup.find_all("tr"):
        cells = row.find_all("td")

        # Datove radky maji 7 bunek
        if len(cells) != 7:
            continue

        position = cells[0].get_text(strip=True)

        lat = float(
            cells[2]
            .get_text(strip=True)
            .split("°")[0]
            .replace(",", ".")
        )
        long = float(
            cells[4]
            .get_text(strip=True)
            .split("°")[0]
            .replace(",", ".")
        )
        height = float(
            cells[6]
            .get_text(strip=True)
            .replace("\xa0", "")
            .replace(",", ".")
        )
        positions.append(position)
        lats.append(lat)
        longs.append(long)
        heights.append(height)

    # Vytvořit výstupní slovník
    data = {
        'positions': positions,
        'lats': lats,
        'longs': longs,
        'heights': heights
    }

    return data


if __name__ == "__main__":
    # Numerický výpočet euklidovské vzdálenosti
    results = distance(
                np.array([[0, 0], [0, 0], [2, 2]]), 
                np.array([[1, 1], [2, 2], [5, 6]])
                )
    print(results)

    # Generování grafu s různými koeficienty
    generate_graph([7, 4, 3], show_figure=False, save_path="part01/figures/vystupni_graf.png")

    # Pokročilá vizualizace sinusového signálu
    generate_sinus(show_figure=False, save_path="part01/figures/sinus_graph.png")

    # Stažení tabulky
    data = download_data()
    print(data)
