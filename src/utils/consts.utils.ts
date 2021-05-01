export const HOUSES_CSV_URL =
  "http://127.0.0.1:8080/house-prices-predictions/src/data/kc_house_data.csv";
export const HOUSE_LABEL_NAME = "price";
export const LABELS_TO_IGNORE = ["id", "date", "price"];
export const modelStatusEnum = {
  0: "not initalized",
  1: "initalized",
  2: "trained",
  3: "saved",
};

export const storageID = `kc-house-price-regression`;
