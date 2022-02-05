from gen_chair.image_scraper import image_scraper

if __name__ == '__main__':
    # keys = ['sitting on chair', 'sitting in chair', 'sitting']
    keys = ['cartoon sitting','stick figure sitting']
    downloader = image_scraper()
    for key in keys:
        downloader.download(key,limit=20, main_directory='datass/')