from unittest import TestCase

import pandas as pd

from src.sepe import parse_report_links_from_listing, parse_sepe_report_html


REPORT_HTML = """
<html>
<body>
<h2>CNO-1111: Miembros del poder ejecutivo Diciembre 2025</h2>
<div class="se-databanner">
  <h4 class="se-databanner--title">Contratos en esta ocupación</h4>
  <p class="se-databanner--figure"><span class="se-databanner--digit">8</span><span>contratos</span></p>
  <p class="se-databanner--figure"><span class="se-databanner--digit">7</span><span>personas</span></p>
</div>
<table>
<caption>Parados según sexo y edad</caption>
<tr><th></th><th>Total</th><th>Variación (1)</th></tr>
<tr><td>Por sexo</td></tr>
<tr><td>Hombre</td><td>24</td><td>-4,00 %</td></tr>
<tr><td>Mujer</td><td>35</td><td>0,00 %</td></tr>
<tr><td>Total</td><td>59</td><td>-1,67 %</td></tr>
<tr><td>Por tramos de edad</td></tr>
<tr><td>18-24</td><td>3</td><td>0,00 %</td></tr>
<tr><td>Total</td><td>59</td><td>-1,67 %</td></tr>
</table>
<table>
<caption>Distribución geográfica de contratos</caption>
<tr><th>Provincia</th><th>Contratos</th><th>Mensual</th></tr>
<tr><td>Barcelona</td><td>2</td><td>0,00 %</td></tr>
<tr><td>Burgos</td><td>3</td><td>0,00 %</td></tr>
</table>
<table>
<caption>Movilidad geográfica de la contratación</caption>
<tr><td></td><td></td><th>Variación</th></tr>
<tr><td>Nº de contratos que permanecen</td><td>3</td><td>-57,14 %</td></tr>
<tr><td>Nº de contratos que se mueven</td><td>5</td><td>0,00 %</td></tr>
</table>
<script>
function drawStuffMovilidad() {
  var dataBar = new google.visualization.DataTable();
  dataBar.addColumn('string', '');
  dataBar.addColumn('number', 'Hombre');
  dataBar.addColumn('number', 'Mujer');
  dataBar.addRow(['Permanecen', 2, 1]);
  dataBar.addRow(['Se mueven', 3, 2]);
}
</script>
</body>
</html>
"""


class SepeTests(TestCase):
    def test_parse_report_links_from_listing_keeps_monthly_links(self):
        html = """
        <h3>CNO-1111: Miembros del poder ejecutivo</h3>
        <table>
        <tr><td>Diciembre</td><td>2025</td>
        <td><a href="/HomeSepe/que-es-observatorio/informacion-mt-por-ocupacion/informacion-mercado-trabajo-por-ocupacion~_mensuales_2025_12_1111-title~.html">Enlace</a></td></tr>
        <tr><td>Anual</td><td>2025</td>
        <td><a href="/HomeSepe/que-es-observatorio/informacion-mt-por-ocupacion/informacion-mercado-trabajo-por-ocupacion-anual~_anuales_2025_1111-title~.html">Enlace</a></td></tr>
        </table>
        """
        links = parse_report_links_from_listing(html, "1111")

        self.assertEqual(len(links), 1)
        self.assertEqual(links[0].period, "2025-12")
        self.assertEqual(links[0].occupation_title, "Miembros del poder ejecutivo")

    def test_parse_report_html_outputs_long_levels(self):
        rows = parse_sepe_report_html(REPORT_HTML, "https://example.test/_mensuales_2025_12_1111-x.html")
        df = pd.DataFrame(rows)

        persons = df[(df["measure"] == "personas") & (df["dimension"] == "total")].iloc[0]
        self.assertEqual(persons["value"], 7)

        women = df[(df["measure"] == "parados") & (df["dimension"] == "gender") & (df["category"] == "Mujer")].iloc[0]
        self.assertEqual(women["value"], 35)
        self.assertEqual(women["gender"], "Mujer")

        province_total = df[
            (df["measure"] == "contratos") & (df["dimension"] == "province") & (df["category"] == "Total")
        ].iloc[0]
        self.assertEqual(province_total["value"], 5)

        mobility_women = df[
            (df["dimension"] == "geographic_mobility") & (df["category"] == "Se mueven") & (df["gender"] == "Mujer")
        ].iloc[0]
        self.assertEqual(mobility_women["value"], 2)
