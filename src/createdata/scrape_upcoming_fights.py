import os
import csv
from typing import List, Dict
from datetime import datetime
from pathlib import Path

from src.createdata.utils import make_soup, print_progress

# Ensure data directory
BASE_PATH = Path(os.getcwd()) / "data"
BASE_PATH.mkdir(parents=True, exist_ok=True)
UPCOMING_FIGHTS_PATH = r"C:\Users\lewis\repos\UFC-Predictions\src\data\upcoming_fights.csv"

class UpcomingFightScraper:
    def __init__(self):
        self.upcoming_events_url = "http://ufcstats.com/statistics/events/upcoming"

    def _get_upcoming_event_links(self) -> List[str]:
        print("Scraping upcoming event links...")
        soup = make_soup(self.upcoming_events_url)
        event_links = []

        for link in soup.find_all("td", {"class": "b-statistics__table-col"}):
            for href in link.find_all("a"):
                event_links.append(href.get("href"))

        return event_links

    def _scrape_event_fights(self, event_link: str) -> List[Dict]:
        soup = make_soup(event_link)
        fights = []

        # Extract event date
        event_date = ""
        for li in soup.find_all("li", {"class": "b-list__box-list-item"}):
            if "Date:" in li.text:
                event_date = li.text.replace("Date:", "").strip()
                try:
                    event_date = datetime.strptime(event_date, "%B %d, %Y").strftime("%Y-%m-%d")
                except Exception:
                    pass

        rows = soup.find_all(
            "tr",
            {"class": "b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click"}
        )

        for row in rows:
            try:
                cols = row.find_all("td")
                if len(cols) < 7:
                    continue

                # Extract red and blue fighters from two <p> tags inside second <td>
                fighter_ps = cols[1].find_all("p")
                if len(fighter_ps) < 2:
                    continue

                red_fighter = fighter_ps[0].get_text(strip=True)
                blue_fighter = fighter_ps[1].get_text(strip=True)

                # Extract weight class from 7th <td>
                weight_class_p = cols[6].find("p")
                weight_class = weight_class_p.get_text(strip=True).replace("\n", " ") if weight_class_p else ""

                if red_fighter and blue_fighter and weight_class:
                    fights.append({
                        "R_fighter": red_fighter,
                        "B_fighter": blue_fighter,
                        "date": event_date,
                        "Fight_type": weight_class
                    })
            except Exception as e:
                print(f"Error parsing row: {e}")
                continue

        return fights

    def scrape_and_save(self):
        event_links = self._get_upcoming_event_links()
        all_fights = []

        print(f"Found {len(event_links)} upcoming events.")
        print_progress(0, len(event_links), prefix="Progress:", suffix="Complete")

        for idx, event_link in enumerate(event_links):
            fights = self._scrape_event_fights(event_link)
            all_fights.extend(fights)
            print_progress(idx + 1, len(event_links), prefix="Progress:", suffix="Complete")

        if all_fights:
            keys = all_fights[0].keys()
            with open(UPCOMING_FIGHTS_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(all_fights)
            print(f"\n✅ Saved {len(all_fights)} upcoming fights to {UPCOMING_FIGHTS_PATH}")
        else:
            print("\n⚠️ No fights found (cards may not yet be posted for upcoming events).")

if __name__ == "__main__":
    scraper = UpcomingFightScraper()
    scraper.scrape_and_save()
