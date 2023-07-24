Release Notes
-------------
**Future Releases**
    * Enhancements
        * Added repo specific token for workflows :pr:`2`
        * PDM Packaging ready for deployment :pr:`2`
        * Added testing workflow for pytest :pr:`2`
    * Fixes
    * Changes
        * Irrelevant workflows removed (`minimum_dependency_checker`) :pr:`2`
    * Documentation Changes
        * Documentation refactored to now fit CheckMate :pr:`2`
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

* *First Release*
