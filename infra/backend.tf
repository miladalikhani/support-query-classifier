terraform {
  backend "gcs" {
    bucket = "datatonic-496102-tfstate"
    prefix = "terraform/state"
  }
}
