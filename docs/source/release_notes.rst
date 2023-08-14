Release Notes
-------------
**Future Releases**
    * Enhancements
        * Added all datachecks except `invalid_target_data_check`` along with tests and utils, migrated over from `EvalML` :pr:`15`
    * Fixes
    * Changes
    * Documentation Changes
        * Updated readme.md, contrubuting.md, and releases.md to reflect CheckMates package installation, quickstart, and useful links :pr:`13`
    * Testing Changes

**v0.1.0 July 28, 2023**
    * Enhancements
        * updated pyproject to v0.1.0 for first release and added project urls :pr:`8`
        * added pdm.lock and .python-version to .gitignore :pr:`8`
        * Added repo specific token for workflows :pr:`2`
        * PDM Packaging ready for deployment :pr:`2`
        * Added testing workflow for pytest :pr:`2`
        * Transfer over base `Data Checks` and `IDColumnData Checks` from the `EvalML` repo :pr:`1`
        * Added in github workflows that are relevant to `DataChecks`, from `EvalML` repository, and modified to fit `DataChecks` wherever possible :pr:`1`
        * Implemented linters and have them successfully running :pr:`1`
    * Fixes
        * Cleanup files and add release workflow :pr:`6`
        * Fixed pytest failures :pr:`1`
        * Workflows are now up and running properly :pr:`1`
    * Changes
        * Irrelevant workflows removed (`minimum_dependency_checker`) :pr:`2`
        * Removed all `EvalML` dependencies and unnecessary functions/comments from `utils`, `tests`, `exceptions`, and `datachecks` :pr:`1`
        * Updated comments to reflect `DataChecks` repository :pr:`1`
        * Restructured file directory to categorize data checks between `datacheck_meta` and `checks` :pr:`1`
        * Restructured pdm packaging to only be relevant to `DataChecks`, now to be renamed to `CheckMate` :pr:`1`
    * Documentation Changes
        * Documentation refactored to now fit `CheckMates` :pr:`11`
        * Documentation refactored to now fit `Checkers` :pr:`4`
        * Documentation refactored to now fit `CheckMate` :pr:`2`
    * Testing Changes
        * Automated testing within github actions :pr:`2`
        * Removed integration testing due to irrelevance with `datacheck_meta` and `checks` :pr:`1`

**v0.0.2 July 26, 2023**
    * Enhancements
        * Added repo specific token for workflows :pr:`2`
        * PDM Packaging ready for deployment :pr:`2`
        * Added testing workflow for pytest :pr:`2`
    * Changes
        * Irrelevant workflows removed (`minimum_dependency_checker`) :pr:`2`
    * Documentation Changes
        * Documentation refactored to now fit CheckMate :pr:`2`
        * Documentation refactored to now fit `Checkers` :pr:`4`
    * Testing Changes
        * Automated testing within github actions :pr:`2`

**v0.0.1 July 18, 2023**
    * Enhancements
        * Transfer over base `Data Checks` and `IDColumnData Checks` from the `EvalML` repo :pr:`1`
        * Added in github workflows that are relevant to `DataChecks`, from `EvalML` repository, and modified to fit `DataChecks` wherever possible :pr:`1`
        * Implemented linters and have them successfully running :pr:`1`
    * Fixes
        * Fixed pytest failures :pr:`1`
        * Workflows are now up and running properly :pr:`1`
    * Changes
        * Removed all `EvalML` dependencies and unnecessary functions/comments from `utils`, `tests`, `exceptions`, and `datachecks` :pr:`1`
        * Updated comments to reflect `DataChecks` repository :pr:`1`
        * Restructured file directory to categorize data checks between `datacheck_meta` and `checks` :pr:`1`
        * Restructured pdm packaging to only be relevant to `DataChecks`, now to be renamed to `CheckMate` :pr:`1`
    * Testing Changes
        * Removed integration testing due to irrelevance with `datacheck_meta` and `checks` :pr:`1`

**v0.0.0 July 3, 2023**
    * *GitHub Repo Created*
