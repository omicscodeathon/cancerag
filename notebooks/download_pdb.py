import marimo

__generated_with = "0.11.12"
app = marimo.App(width="full", app_title="PDB File Download")


@app.cell
def _():
    import os
    import requests
    import json
    import time
    from tqdm import tqdm  # For progress bars

    def search_pdb_structures(query_text, return_count=500):
        """
        Search the PDB for structures matching the query text using the current API format.

        Args:
            query_text (str): The search query
            return_count (int): Maximum number of results to return

        Returns:
            list: List of PDB IDs matching the query
        """
        # The RCSB PDB search API endpoint
        search_url = "https://search.rcsb.org/rcsbsearch/v2/query"

        # Updated query structure based on current API requirements
        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entity_source_organism.taxonomy_lineage.name",
                            "operator": "contains_phrase",
                            "value": "Homo sapiens"
                        }
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "struct.title",
                            "operator": "contains_words",
                            "value": query_text
                        }
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.structure_determination_methodology",
                            "operator": "exact_match",
                            "value": "experimental"
                        }
                    }
                ]
            },
            "return_type": "entry",
            "request_options": {
                "scoring_strategy": "combined",
                "sort": [{"sort_by": "score", "direction": "desc"}],
                # "pager": {
                #     "start": 0,
                #     "rows": return_count
                # }
            }
        }

        # Make the POST request to the search API
        response = requests.post(search_url, json=query)

        # Check if the request was successful
        if response.status_code == 200:
            result_json = response.json()
            pdb_ids = [hit["identifier"] for hit in result_json.get("result_set", [])]
            return pdb_ids
        else:
            print(f"Search request failed with status code {response.status_code}")
            print(response.text)
            return []

    def get_structure_metadata(pdb_id):
        """
        Get metadata about a PDB structure to categorize it correctly.

        Args:
            pdb_id (str): The PDB ID

        Returns:
            dict: Structure metadata including receptor type
        """
        # The RCSB PDB data API endpoint for polymer entities (proteins)
        info_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"

        # Make the GET request to the data API
        response = requests.get(info_url)

        # Default classification
        metadata = {
            "id": pdb_id,
            "type": "unknown",
            "title": "",
            "resolution": None
        }

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()

            # Extract useful information
            title = data.get("struct", {}).get("title", "").lower()
            metadata["title"] = title

            # Try to get resolution
            resolution = data.get("refine", [{}])[0].get("ls_d_res_high") if data.get("refine") else None
            metadata["resolution"] = resolution

            # Classify based on title
            if "alpha" in title or "er-alpha" in title or "eralpha" in title or "erα" in title:
                metadata["type"] = "ER-alpha"
            elif "beta" in title or "er-beta" in title or "erbeta" in title or "erβ" in title:
                metadata["type"] = "ER-beta"
            elif "estrogen receptor" in title:
                metadata["type"] = "ER-complex"

            return metadata
        else:
            print(f"Data request for {pdb_id} failed with status code {response.status_code}")
            return metadata

    def download_pdb_file(pdb_id, output_dir):
        """
        Download a PDB file by its ID.

        Args:
            pdb_id (str): The PDB ID
            output_dir (str): Directory to save the file

        Returns:
            str: Path to the downloaded file or None if download fails
        """
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Define the output file path
        output_file = os.path.join(output_dir, f"{pdb_id}.pdb")

        # If the file already exists, don't download it again
        if os.path.exists(output_file):
            print(f"File {output_file} already exists. Skipping download.")
            return output_file

        # The RCSB PDB download API endpoint
        download_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

        # Make the GET request to download the file
        response = requests.get(download_url)

        # Check if the request was successful
        if response.status_code == 200:
            # Write the file to disk
            with open(output_file, "wb") as f:
                f.write(response.content)
            return output_file
        else:
            print(f"Download request for {pdb_id} failed with status code {response.status_code}")
            return None

    def download_receptor_structures(output_dir="data/pdb", max_downloads=100):
        """
        Search for and download estrogen receptor structures.

        Args:
            output_dir (str): Directory to save the files
            max_downloads (int): Maximum number of structures to download

        Returns:
            dict: Statistics about downloaded structures
        """
        # Create a dictionary to store metadata for each receptor type
        receptor_searches = [
            {"search_term": "estrogen receptor alpha", "type": "ER-alpha"},
            {"search_term": "estrogen receptor beta", "type": "ER-beta"},
            {"search_term": "estrogen receptor", "type": "ER-complex"}
        ]

        # Search for each receptor type
        all_ids = set()
        results_by_type = {"ER-alpha": [], "ER-beta": [], "ER-complex": []}

        for search in receptor_searches:
            print(f"Searching for {search['type']} structures...")
            pdb_ids = search_pdb_structures(search["search_term"])

            print(f"Found {len(pdb_ids)} potential {search['type']} structures")
            all_ids.update(pdb_ids)

        # Limit the total number of downloads
        all_ids = list(all_ids)[:max_downloads]
        print(f"Found {len(all_ids)} unique structures to process")

        # Get metadata and categorize each structure more precisely
        classified_structures = {"ER-alpha": [], "ER-beta": [], "ER-complex": [], "unknown": []}

        print("Classifying structures based on metadata...")
        for pdb_id in tqdm(all_ids, desc="Retrieving metadata"):
            # Add a small delay to avoid overloading the server
            time.sleep(0.2)

            metadata = get_structure_metadata(pdb_id)
            classified_structures[metadata["type"]].append(metadata)

        # Print classification results
        for receptor_type, structures in classified_structures.items():
            if receptor_type != "unknown":
                print(f"Classified {len(structures)} structures as {receptor_type}")

        # Download the structures
        downloaded_files = {"ER-alpha": [], "ER-beta": [], "ER-complex": [], "unknown": []}

        for receptor_type, structures in classified_structures.items():
            if structures:
                print(f"\nDownloading {receptor_type} structures...")

                for metadata in tqdm(structures, desc=f"Downloading {receptor_type}"):
                    # Add a small delay to avoid overloading the server
                    time.sleep(0.5)

                    # Create type-specific subdirectory
                    type_dir = os.path.join(output_dir, receptor_type.lower().replace("-", "_"))

                    file_path = download_pdb_file(metadata["id"], type_dir)
                    if file_path is not None:
                        downloaded_files[receptor_type].append({
                            "id": metadata["id"],
                            "path": file_path,
                            "title": metadata["title"],
                            "resolution": metadata["resolution"]
                        })

        # Count total downloads
        total_downloads = sum(len(files) for files in downloaded_files.values())
        print(f"\nDownloaded {total_downloads} PDB files to {output_dir}")

        # Create and save a summary file
        summary = {
            "total_structures": total_downloads,
            "structures_by_type": {
                k: {"count": len(v), "structures": v} 
                for k, v in downloaded_files.items() if k != "unknown" and v
            }
        }

        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    if __name__ == "__main__":
        # Set the output directory
        output_dir = "../data/pdb"

        # Run the download function
        summary = download_receptor_structures(output_dir=output_dir, max_downloads=100)

        # Print a summary
        print("\nDownload Summary:")
        print(f"Total structures: {summary['total_structures']}")
        for receptor_type, data in summary['structures_by_type'].items():
            print(f"{receptor_type}: {data['count']} structures")
    return (
        data,
        download_pdb_file,
        download_receptor_structures,
        get_structure_metadata,
        json,
        os,
        output_dir,
        receptor_type,
        requests,
        search_pdb_structures,
        summary,
        time,
        tqdm,
    )


if __name__ == "__main__":
    app.run()
